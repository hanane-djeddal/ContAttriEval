import unsloth
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import datasets
import json
import os
import random
from unsloth import FastLanguageModel
from unsloth import tokenizer_utils

# Disable unsloth's token fixing if needed
def do_nothing(*args, **kwargs):
    pass
tokenizer_utils.fix_untrained_tokens = do_nothing

os.environ['HF_HOME'] = os.environ.get('WORK', '.') + '/.cache/huggingface'
os.environ['WANDB_MODE'] = 'offline'

IGNORE_INDEX = -100

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Qwen3-4B model"}
    )
    max_qwen_length: int = field(default=4096)
    load_in_4bit: bool = field(default=False)
    dtype: Optional[str] = field(default=None)

@dataclass
class LoRAArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    use_rslora: bool = field(default=True)
    use_gradient_checkpointing: str = field(default="unsloth")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to training data"})
    dev_data_path: str = field(default=None, metadata={"help": "Path to dev data"})
    dataset_version: str = field(default="v3.0", metadata={"help": "Dataset version"})
    use_contrastive: bool = field(default=True, metadata={"help": "Use contrastive learning"})
    num_positives: int = field(default=8, metadata={"help": "Number of positives per anchor"})
    num_negatives: int = field(default=8, metadata={"help": "Number of negatives per anchor"})

@dataclass
class ContrastiveTrainingArguments:
    contrastive_weight: float = field(
        default=0.15,
        metadata={"help": "Weight for contrastive loss (lower to preserve knowledge)"}
    )
    classification_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for classification loss"}
    )
    contrastive_temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature for contrastive loss"}
    )
    contrastive_warmup_steps: int = field(
        default=500,
        metadata={"help": "Warmup steps for contrastive loss"}
    )
    use_in_batch_negatives: bool = field(
        default=True,
        metadata={"help": "Use in-batch negatives when explicit negatives missing"}
    )


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for Qwen3 hidden states.
    Preserves pre-trained knowledge while enhancing representations.
    """
    def __init__(self, temperature=0.07, use_in_batch_negatives=True):
        super().__init__()
        self.temperature = temperature
        self.use_in_batch_negatives = use_in_batch_negatives
        logging.info(f"Contrastive loss initialized with temperature: {self.temperature}")
    
    def forward(self, anchor_embeds, positive_embeds, negative_embeds=None,
                anchor_labels=None, all_batch_embeds=None, all_batch_labels=None):
        """
        Args:
            anchor_embeds: [batch_size, hidden_dim]
            positive_embeds: [batch_size, num_pos, hidden_dim]
            negative_embeds: [batch_size, num_neg, hidden_dim] or None
            anchor_labels: [batch_size] - binary labels (0 or 1)
            all_batch_embeds: [batch_size, hidden_dim] - all anchors in batch
            all_batch_labels: [batch_size] - labels for all anchors
        Returns:
            loss: scalar
        """
        batch_size = anchor_embeds.shape[0]
        device = anchor_embeds.device
        
        # Normalize embeddings (important for stable training)
        anchor_embeds = F.normalize(anchor_embeds, dim=-1)
        positive_embeds = F.normalize(positive_embeds, dim=-1)
        
        anchor_embeds_exp = anchor_embeds.unsqueeze(1)  # [B, 1, D]
        
        # Positive similarities
        pos_sim = torch.sum(anchor_embeds_exp * positive_embeds, dim=-1) / self.temperature
        
        # Determine negative similarities
        use_explicit_negatives = (negative_embeds is not None and 
                                  negative_embeds.numel() > 0 and 
                                  negative_embeds.shape[1] > 0)
        
        if use_explicit_negatives:
            negative_embeds = F.normalize(negative_embeds, dim=-1)
            neg_sim = torch.sum(anchor_embeds_exp * negative_embeds, dim=-1) / self.temperature
            
        elif self.use_in_batch_negatives and all_batch_embeds is not None and anchor_labels is not None:
            # Fallback to in-batch negatives
            all_batch_embeds = F.normalize(all_batch_embeds, dim=-1)
            batch_sim = torch.matmul(anchor_embeds, all_batch_embeds.t()) / self.temperature
            
            # Create negative mask: different labels and not self
            label_diff = (anchor_labels.unsqueeze(1) != all_batch_labels.unsqueeze(0)).float()
            self_mask = 1 - torch.eye(batch_size, device=device)
            neg_mask = label_diff * self_mask
            
            # Extract negative similarities
            neg_sim_list = []
            for i in range(batch_size):
                neg_indices = neg_mask[i].bool()
                if neg_indices.sum() > 0:
                    neg_sim_list.append(batch_sim[i][neg_indices])
                else:
                    neg_sim_list.append(torch.tensor([], device=device))
            
            # Pad to uniform length
            if any(len(n) > 0 for n in neg_sim_list):
                max_neg = max(len(n) for n in neg_sim_list)
                neg_sim_padded = []
                for neg_sim_i in neg_sim_list:
                    if len(neg_sim_i) == 0:
                        neg_sim_i = torch.full((max_neg,), -1e9, device=device)
                    elif len(neg_sim_i) < max_neg:
                        pad_size = max_neg - len(neg_sim_i)
                        padding = torch.full((pad_size,), -1e9, device=device)
                        neg_sim_i = torch.cat([neg_sim_i, padding])
                    neg_sim_padded.append(neg_sim_i)
                neg_sim = torch.stack(neg_sim_padded)
            else:
                neg_sim = None
        else:
            neg_sim = None
        
        # Compute contrastive loss
        num_pos = positive_embeds.shape[1]
        losses = []
        
        for i in range(num_pos):
            numerator = pos_sim[:, i]
            
            if neg_sim is not None:
                all_sim = torch.cat([pos_sim, neg_sim], dim=1)
                denominator = torch.logsumexp(all_sim, dim=1)
            else:
                denominator = torch.logsumexp(pos_sim, dim=1)
            
            loss = -numerator + denominator
            losses.append(loss)
        
        total_loss = torch.stack(losses).mean()
        return total_loss


def get_qwen_hidden_state(model, input_ids, attention_mask):
    """
    Extract hidden state from last non-padding token of Qwen3.
    This preserves the model's natural representation space.
    
    Args:
        model: Qwen3 model
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
    Returns:
        hidden_state: [batch_size, hidden_dim]
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    
    # Get last layer hidden states
    hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
    
    # Get last non-padding token for each sequence
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    
    # Extract last token hidden state
    last_hidden = hidden_states[
        torch.arange(batch_size, device=hidden_states.device),
        sequence_lengths
    ]
    
    return last_hidden
def prepare_qwen_dataset(data_args, split="train"):
    """
    Prepare HuggingFace Dataset with contrastive structure.
    Returns a dataset ready for processing.
    """
    # Qwen3 prompt template
    prompt_template = """You will be given a claim and a document. Determine whether the claim is 'GROUNDED' or 'NOT GROUNDED' based on the document.A 'GROUNDED' claim is fully supported by the information provided in the document. It should be directly verifiable from the document. Only return the classification as the answer: 1 for 'GROUNDED' or 0 for 'NOT GROUNDED' without any explanation.
    CLAIM: {} 
    DOCUMENT: {}
    CLASSIFICATION:"""
    
    def format_example(example):
        """Format example using Qwen3 prompt template"""
        claim = example.get("claim", "")
        if claim in ["nan", "", None]:
            claim = ""
        
        # Concatenate references as document
        documents = example.get("references", [])
        document = "\n\n".join(documents) if documents else ""
        
        # Format prompt (without classification for input)
        prompt = prompt_template.format(claim, document)
        
        # Get label
        label = "1" if str(example.get('attribution_label', '')) == "attributable" else "0"
        
        # Complete text with label for training
        full_text = prompt + label
        
        return prompt, label, full_text
    
    # Load dataset
    data_path = os.environ.get('WORK', '.') + "/AttributionBench"
    use_contrastive = data_args.use_contrastive and split == "train"
    
    if split == "train" and use_contrastive:
        # Load contrastive dataset
        data_path = os.environ.get('WORK', '.') + "/" + data_args.dataset_version
        dataset = datasets.load_from_disk(data_path)[split]
        
        # Process with contrastive structure
        def process_contrastive(example):
            # Check if anchor-only
            has_positives = "positives" in example and len(example["positives"]) > 0
            has_negatives = "negatives" in example and len(example["negatives"]) > 0
            is_anchor_only = not has_positives and not has_negatives
            
            if is_anchor_only:
                anchor = example.get('anchor', example)
                prompt, label, full_text = format_example(anchor)
                return {
                    "text": full_text,
                    "label": label,
                    "positives": [],
                    "negatives": [],
                    "is_anchor_only": True
                }
            
            # Contrastive mode
            anchor = example.get("anchor", example)
            positives = example["positives"]
            negatives = example["negatives"]
            
            # Sample subset
            num_pos = min(data_args.num_positives, len(positives))
            num_neg = min(data_args.num_negatives, len(negatives))
            
            sampled_positives = random.sample(positives, num_pos) if len(positives) > 0 else []
            sampled_negatives = random.sample(negatives, num_neg) if len(negatives) > 0 else []
            
            # Format anchor
            anchor_prompt, anchor_label, anchor_text = format_example(anchor)
            
            # Format positives
            pos_data = []
            for pos in sampled_positives:
                _, _, full_text = format_example(pos)
                pos_data.append(full_text)
            
            # Format negatives
            neg_data = []
            for neg in sampled_negatives:
                _, _, full_text = format_example(neg)
                neg_data.append(full_text)
            
            return {
                "text": anchor_text,
                "label": anchor_label,
                "positives": pos_data,
                "negatives": neg_data,
                "is_anchor_only": False
            }
        
        dataset = dataset.map(process_contrastive, desc="Processing contrastive examples")
        
    else:
        # Load regular dataset
        dataset = datasets.load_from_disk(data_path)
        dataset = dataset[split] if split in dataset else dataset["dev"]
        
        # Process regular examples
        def process_regular(example):
            prompt, label, full_text = format_example(example)
            return {
                "text": full_text,
                "label": label
            }
        
        dataset = dataset.map(process_regular, desc="Processing regular examples")
    
    logging.info(f"Loaded {len(dataset)} examples for {split}")
    if use_contrastive:
        logging.info(f"Contrastive enabled: {data_args.num_positives} pos, {data_args.num_negatives} neg")
    
    return dataset


class QwenContrastiveCollator:
    """
    Custom collator for Qwen3 contrastive learning.
    Handles variable-length positives/negatives.
    """
    def __init__(self, tokenizer, max_seq_length=4096, use_contrastive=True):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_contrastive = use_contrastive
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # Tokenize anchors
        anchor_texts = [inst["text"] for inst in instances]
        anchor_encodings = self.tokenizer(
            anchor_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Extract labels
        anchor_labels_text = [inst["label"] for inst in instances]
        anchor_labels = torch.tensor([int(l) for l in anchor_labels_text])
        
        if not self.use_contrastive:
            return {
                "input_ids": anchor_encodings.input_ids,
                "attention_mask": anchor_encodings.attention_mask,
                "labels": anchor_encodings.input_ids.clone(),
                "binary_labels": anchor_labels
            }
        
        # Check if batch has contrastive data
        has_contrastive = any(
            (isinstance(inst.get("positives", []), list) and len(inst.get("positives", [])) > 0) or
            (isinstance(inst.get("negatives", []), list) and len(inst.get("negatives", [])) > 0)
            for inst in instances
        )
        
        if not has_contrastive:
            return {
                "input_ids": anchor_encodings.input_ids,
                "attention_mask": anchor_encodings.attention_mask,
                "labels": anchor_encodings.input_ids.clone(),
                "binary_labels": anchor_labels
            }
        
        # Collect all positives (now they're just text strings)
        all_pos_texts = []
        pos_counts = []
        for inst in instances:
            positives = inst.get("positives", [])
            if isinstance(positives, list):
                pos_counts.append(len(positives))
                all_pos_texts.extend(positives)
            else:
                pos_counts.append(0)
        
        max_pos = max(pos_counts) if pos_counts else 0
        
        # Tokenize positives
        if max_pos > 0 and len(all_pos_texts) > 0:
            pos_encodings = self.tokenizer(
                all_pos_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            # Reshape to [batch, max_pos, seq_len]
            pos_input_ids_list = []
            pos_attention_mask_list = []
            idx = 0
            for count in pos_counts:
                if count == 0:
                    # Dummy positives
                    dummy_ids = torch.zeros((max_pos, pos_encodings.input_ids.shape[1]), dtype=torch.long)
                    dummy_mask = torch.zeros((max_pos, pos_encodings.attention_mask.shape[1]), dtype=torch.long)
                    pos_input_ids_list.append(dummy_ids)
                    pos_attention_mask_list.append(dummy_mask)
                else:
                    batch_pos_ids = pos_encodings.input_ids[idx:idx+count]
                    batch_pos_mask = pos_encodings.attention_mask[idx:idx+count]
                    
                    if count < max_pos:
                        pad_amount = max_pos - count
                        pad_ids = torch.zeros((pad_amount, batch_pos_ids.shape[1]), dtype=torch.long)
                        pad_mask = torch.zeros((pad_amount, batch_pos_mask.shape[1]), dtype=torch.long)
                        batch_pos_ids = torch.cat([batch_pos_ids, pad_ids], dim=0)
                        batch_pos_mask = torch.cat([batch_pos_mask, pad_mask], dim=0)
                    
                    pos_input_ids_list.append(batch_pos_ids)
                    pos_attention_mask_list.append(batch_pos_mask)
                    idx += count
            
            pos_input_ids = torch.stack(pos_input_ids_list)
            pos_attention_mask = torch.stack(pos_attention_mask_list)
        else:
            pos_input_ids = None
            pos_attention_mask = None
        
        # Collect all negatives (now they're just text strings)
        all_neg_texts = []
        neg_counts = []
        for inst in instances:
            negatives = inst.get("negatives", [])
            if isinstance(negatives, list):
                neg_counts.append(len(negatives))
                all_neg_texts.extend(negatives)
            else:
                neg_counts.append(0)
        
        max_neg = max(neg_counts) if neg_counts else 0
        
        # Tokenize negatives
        if max_neg > 0 and len(all_neg_texts) > 0:
            neg_encodings = self.tokenizer(
                all_neg_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            neg_input_ids_list = []
            neg_attention_mask_list = []
            idx = 0
            for count in neg_counts:
                if count == 0:
                    dummy_ids = torch.zeros((max_neg, neg_encodings.input_ids.shape[1]), dtype=torch.long)
                    dummy_mask = torch.zeros((max_neg, neg_encodings.attention_mask.shape[1]), dtype=torch.long)
                    neg_input_ids_list.append(dummy_ids)
                    neg_attention_mask_list.append(dummy_mask)
                else:
                    batch_neg_ids = neg_encodings.input_ids[idx:idx+count]
                    batch_neg_mask = neg_encodings.attention_mask[idx:idx+count]
                    
                    if count < max_neg:
                        pad_amount = max_neg - count
                        pad_ids = torch.zeros((pad_amount, batch_neg_ids.shape[1]), dtype=torch.long)
                        pad_mask = torch.zeros((pad_amount, batch_neg_mask.shape[1]), dtype=torch.long)
                        batch_neg_ids = torch.cat([batch_neg_ids, pad_ids], dim=0)
                        batch_neg_mask = torch.cat([batch_neg_mask, pad_mask], dim=0)
                    
                    neg_input_ids_list.append(batch_neg_ids)
                    neg_attention_mask_list.append(batch_neg_mask)
                    idx += count
            
            neg_input_ids = torch.stack(neg_input_ids_list)
            neg_attention_mask = torch.stack(neg_attention_mask_list)
        else:
            neg_input_ids = None
            neg_attention_mask = None
        
        return {
            "input_ids": anchor_encodings.input_ids,
            "attention_mask": anchor_encodings.attention_mask,
            "labels": anchor_encodings.input_ids.clone(),
            "binary_labels": anchor_labels,
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_attention_mask,
            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_attention_mask,
        }


class QwenContrastiveTrainer(SFTTrainer):
    """
    Custom trainer combining classification + contrastive learning for Qwen3.
    Preserves pre-trained knowledge through careful loss balancing.
    """
    def __init__(
        self, 
        *args,
        contrastive_loss_fn=None,
        contrastive_weight=0.15,
        classification_weight=1.0,
        contrastive_warmup_steps=500,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.contrastive_loss_fn = contrastive_loss_fn
        self.base_contrastive_weight = contrastive_weight
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        self.contrastive_warmup_steps = contrastive_warmup_steps
        
        logging.info(f"Contrastive Trainer initialized:")
        logging.info(f"  - Base contrastive weight: {self.base_contrastive_weight}")
        logging.info(f"  - Classification weight: {self.classification_weight}")
        logging.info(f"  - Warmup steps: {self.contrastive_warmup_steps}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined loss: classification + contrastive.
        Implements warmup for contrastive weight.
        """
        # Apply contrastive warmup
        if self.state.global_step < self.contrastive_warmup_steps:
            warmup_factor = self.state.global_step / self.contrastive_warmup_steps
            self.contrastive_weight = self.base_contrastive_weight * warmup_factor
        else:
            self.contrastive_weight = self.base_contrastive_weight
        
        # Extract contrastive inputs
        pos_input_ids = inputs.pop("pos_input_ids", None)
        pos_attention_mask = inputs.pop("pos_attention_mask", None)
        neg_input_ids = inputs.pop("neg_input_ids", None)
        neg_attention_mask = inputs.pop("neg_attention_mask", None)
        binary_labels = inputs.pop("binary_labels", None)
        
        # Forward pass for classification
        outputs = model(**inputs)
        classification_loss = outputs.loss
        
        # If no contrastive data, return only classification loss
        if pos_input_ids is None or pos_input_ids.numel() == 0 or pos_input_ids.shape[1] == 0:
            return (classification_loss, outputs) if return_outputs else classification_loss
        
        # Extract hidden states for contrastive learning
        anchor_hidden = get_qwen_hidden_state(
            model,
            inputs["input_ids"],
            inputs["attention_mask"]
        )
        
        # Get positives hidden states
        batch_size, num_pos, seq_len = pos_input_ids.shape
        pos_input_ids_flat = pos_input_ids.view(-1, seq_len)
        pos_attention_mask_flat = pos_attention_mask.view(-1, seq_len)
        
        pos_hidden_flat = get_qwen_hidden_state(
            model,
            pos_input_ids_flat,
            pos_attention_mask_flat
        )
        pos_hidden = pos_hidden_flat.view(batch_size, num_pos, -1)
        
        # Get negatives hidden states (if available)
        neg_hidden = None
        if neg_input_ids is not None and neg_input_ids.numel() > 0 and neg_input_ids.shape[1] > 0:
            batch_size, num_neg, seq_len = neg_input_ids.shape
            neg_input_ids_flat = neg_input_ids.view(-1, seq_len)
            neg_attention_mask_flat = neg_attention_mask.view(-1, seq_len)
            
            neg_hidden_flat = get_qwen_hidden_state(
                model,
                neg_input_ids_flat,
                neg_attention_mask_flat
            )
            neg_hidden = neg_hidden_flat.view(batch_size, num_neg, -1)
        
        # Compute contrastive loss
        contrastive_loss = self.contrastive_loss_fn(
            anchor_embeds=anchor_hidden,
            positive_embeds=pos_hidden,
            negative_embeds=neg_hidden,
            anchor_labels=binary_labels,
            all_batch_embeds=anchor_hidden,
            all_batch_labels=binary_labels
        )
        
        # Combined loss
        total_loss = (
            self.classification_weight * classification_loss +
            self.contrastive_weight * contrastive_loss
        )
        
        # Logging
        if self.state.global_step % 10 == 0:
            has_explicit_neg = neg_hidden is not None
            logging.info(
                f"Step {self.state.global_step}: "
                f"Total={total_loss:.4f}, "
                f"Class={classification_loss:.4f}, "
                f"Contr={contrastive_loss:.4f} "
                f"(weight={self.contrastive_weight:.4f}, explicit_neg={has_explicit_neg})"
            )
        
        return (total_loss, outputs) if return_outputs else total_loss



def compute_metrics(eval_preds):
    """Compute accuracy for evaluation"""
    logits = eval_preds.predictions[0] if isinstance(eval_preds.predictions, tuple) else eval_preds.predictions
    preds = np.argmax(logits, axis=-1)
    labels = eval_preds.label_ids
    
    # Handle padding
    labels = np.where(labels != -100, labels, 0)
    
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


def train():
    """Main training function"""
    # Parse arguments
    parser = transformers.HfArgumentParser((
        ModelArguments,
        LoRAArguments,
        DataArguments,
        ContrastiveTrainingArguments,
        SFTConfig
    ))
    model_args, lora_args, data_args, contrastive_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Load model with Unsloth
    model_name = os.environ['WORK'] + '/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c' #"Qwen/Qwen3-4B";
    # model_args.model_name_or_path or (os.environ.get('WORK', '.') + '/.cache/huggingface/hub/models--Qwen--Qwen2.5-4B')
    dtype = model_args.dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    load_in_4bit = False
    
    logging.info(f"Loading model: {model_name}")
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=model_args.max_qwen_length,
        dtype=dtype,
        load_in_4bit=model_args.load_in_4bit,
    )


    # Apply LoRA
    logging.info("Applying LoRA adapters")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_args.lora_r,
        target_modules=[
            "lm_head",
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=lora_args.use_gradient_checkpointing,
        random_state=3407,
        use_rslora=lora_args.use_rslora,
    )
    
    # Create datasets
    logging.info("Loading datasets")
    train_dataset = prepare_qwen_dataset(
        data_args=data_args,
        split="train"
    )
    
    eval_data_args = copy.deepcopy(data_args)
    eval_data_args.use_contrastive = False
    
    eval_dataset = prepare_qwen_dataset(
        data_args=eval_data_args,
        split="dev"
    )
    
    # Create collators
    train_collator = QwenContrastiveCollator(
        tokenizer=tokenizer,
        max_seq_length=model_args.max_qwen_length,
        use_contrastive=data_args.use_contrastive
    )
    
    eval_collator = QwenContrastiveCollator(
        tokenizer=tokenizer,
        max_seq_length=model_args.max_qwen_length,
        use_contrastive=False
    )
    

    # Initialize contrastive loss
    contrastive_loss_fn = SupervisedContrastiveLoss(
        temperature=contrastive_args.contrastive_temperature,
        use_in_batch_negatives=contrastive_args.use_in_batch_negatives
    )
    
    logging.info(f"GPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    for name, module in model.named_modules():
       print(name, type(module))
    # Create trainer
    trainer = QwenContrastiveTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_collator,
        args=training_args,
        contrastive_loss_fn=contrastive_loss_fn,
        contrastive_weight=contrastive_args.contrastive_weight,
        classification_weight=contrastive_args.classification_weight,
        contrastive_warmup_steps=contrastive_args.contrastive_warmup_steps,
        #**data_module,
    )
    
    # Train
    logging.info("Starting training...")
    trainer.train()
    
    logging.info(f"GPU memory after training: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Save model
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    logging.info(f"Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    train()