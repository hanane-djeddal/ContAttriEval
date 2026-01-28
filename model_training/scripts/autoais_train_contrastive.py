import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer
from datasets import load_dataset, Features, Value
import datasets
import json
import os
import random

os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
os.environ['WANDB_MODE'] = 'offline'

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

T5_TYPES=["claim_erronous_change","claim_numerical_mismatch","modify_passage-add_relevant_to_claim","claim_combine_facts","claim_add_to_the_claim_contradicting_info","modify_passage-add_contradiction","modify_passage-add_conflicting_sources","claim_infer_claim","claim_over_infer_claim"]

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/flan-t5-xl")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    dev_data_path: str = field(default=None, metadata={"help": "Path to the dev data."})
    dataset_version: str = field(
        default="v3.0",
        metadata={"help": "Dataset version"}
    )
    error_type: str = field(default=None)
    template: str = field(default="base_c_e")
    template_path: str = field(default="src/train/template.json")
    use_contrastive: bool = field(default=True, metadata={"help": "Use contrastive learning"})
    num_positives: int = field(default=8, metadata={"help": "Number of positives to sample per anchor"})
    num_negatives: int = field(default=8, metadata={"help": "Number of negatives to sample per anchor"})
    include_anchor_only: bool = field(
        default=True, 
        metadata={"help": "Include examples with no positives/negatives (anchor-only)"}
    )    
    filter_error_types: bool = field(
        default=False, 
        metadata={"help": "Filter error types only leaving select ones"}
    )

@dataclass
class Seq2SeqTrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    contrastive_weight: float = field(
        default=0.5, 
        metadata={"help": "Weight for contrastive loss (lambda1)"}
    )
    classification_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for classification loss (lambda2)"}
    )
    contrastive_temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature for contrastive loss"}
    )
    use_decoder_embedding: bool = field(
        default=True,
        metadata={"help": "Use decoder's hidden state (True) or encoder's first token (False) for contrastive learning"}
    )
    use_in_batch_negatives: bool = field(
        default=True,
        metadata={"help": "Use in-batch negatives when explicit negatives are missing"}
    )


# class SupervisedContrastiveLoss(nn.Module):
#     """
#     Supervised Contrastive Loss for T5 hidden states.
#     Pulls positives closer, pushes negatives away.
#     """
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature
    
#     def forward(self, anchor_embeds, positive_embeds, negative_embeds):
#         """
#         Args:
#             anchor_embeds: [batch_size, hidden_dim]
#             positive_embeds: [batch_size, num_pos, hidden_dim]
#             negative_embeds: [batch_size, num_neg, hidden_dim]
#         Returns:
#             loss: scalar
#         """
#         batch_size = anchor_embeds.shape[0]
        
#         # Normalize embeddings
#         anchor_embeds = F.normalize(anchor_embeds, dim=-1)  # [B, D]
#         positive_embeds = F.normalize(positive_embeds, dim=-1)  # [B, N_pos, D]
#         negative_embeds = F.normalize(negative_embeds, dim=-1)  # [B, N_neg, D]
        
#         # Compute similarities
#         # anchor_embeds: [B, D] -> [B, 1, D]
#         anchor_embeds_exp = anchor_embeds.unsqueeze(1)
        
#         # Positive similarities: [B, N_pos]
#         pos_sim = torch.sum(anchor_embeds_exp * positive_embeds, dim=-1) / self.temperature
        
#         # Negative similarities: [B, N_neg]
#         neg_sim = torch.sum(anchor_embeds_exp * negative_embeds, dim=-1) / self.temperature
        
#         # Concatenate: [B, N_pos + N_neg]
#         all_sim = torch.cat([pos_sim, neg_sim], dim=1)
        
#         # Create labels: positives are first N_pos indices
#         num_pos = positive_embeds.shape[1]
        
#         # Compute loss for each positive
#         losses = []
#         for i in range(num_pos):
#             # LogSumExp over all similarities
#             logits = all_sim  # [B, N_pos + N_neg]
            
#             # Numerator: exp(sim with this positive)
#             numerator = pos_sim[:, i]  # [B]
            
#             # Denominator: sum of exp(all similarities)
#             denominator = torch.logsumexp(all_sim, dim=1)  # [B]
            
#             # Loss: -log(exp(pos_i) / sum(exp(all)))
#             loss = -numerator + denominator
#             losses.append(loss)
        
#         # Average over positives and batch
#         total_loss = torch.stack(losses).mean()
        
#         return total_loss

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for T5 hidden states.
    Pulls positives closer, pushes negatives away.
    Supports in-batch negatives when explicit negatives are missing.
    """
    def __init__(self, temperature=0.07, use_in_batch_negatives=True):
        super().__init__()
        self.temperature = temperature
        self.use_in_batch_negatives = use_in_batch_negatives
        print("Setting temp to:", self.temperature )
    
    def forward(self, anchor_embeds, positive_embeds, negative_embeds=None,
                anchor_labels=None, all_batch_embeds=None, all_batch_labels=None):
        """
        Args:
            anchor_embeds: [batch_size, hidden_dim]
            positive_embeds: [batch_size, num_pos, hidden_dim]
            negative_embeds: [batch_size, num_neg, hidden_dim] or None
            anchor_labels: [batch_size] - binary labels (0 or 1) for anchors
            all_batch_embeds: [batch_size, hidden_dim] - all anchor embeddings in batch
            all_batch_labels: [batch_size] - labels for all anchors in batch
        Returns:
            loss: scalar
        """
        batch_size = anchor_embeds.shape[0]
        device = anchor_embeds.device
        
        # Normalize embeddings
        anchor_embeds = F.normalize(anchor_embeds, dim=-1)  # [B, D]
        positive_embeds = F.normalize(positive_embeds, dim=-1)  # [B, N_pos, D]
        
        anchor_embeds_exp = anchor_embeds.unsqueeze(1)  # [B, 1, D]
        
        # Positive similarities: [B, N_pos]
        pos_sim = torch.sum(anchor_embeds_exp * positive_embeds, dim=-1) / self.temperature
        
        # Determine negative similarities
        use_explicit_negatives = (negative_embeds is not None and 
                                  negative_embeds.numel() > 0 and 
                                  negative_embeds.shape[1] > 0)
        
        if use_explicit_negatives:
            # Use provided negatives
            negative_embeds = F.normalize(negative_embeds, dim=-1)  # [B, N_neg, D]
            neg_sim = torch.sum(anchor_embeds_exp * negative_embeds, dim=-1) / self.temperature
            
        elif self.use_in_batch_negatives and all_batch_embeds is not None and anchor_labels is not None:
            # Fallback to in-batch negatives
            all_batch_embeds = F.normalize(all_batch_embeds, dim=-1)  # [B, D]
            
            # Compute pairwise similarities: [B, B]
            batch_sim = torch.matmul(anchor_embeds, all_batch_embeds.t()) / self.temperature
            
            # Create negative mask: different labels and not self
            label_diff = (anchor_labels.unsqueeze(1) != all_batch_labels.unsqueeze(0)).float()
            self_mask = 1 - torch.eye(batch_size, device=device)
            neg_mask = label_diff * self_mask  # [B, B]
            
            # Extract negative similarities per anchor
            neg_sim_list = []
            for i in range(batch_size):
                neg_indices = neg_mask[i].bool()
                if neg_indices.sum() > 0:
                    neg_sim_list.append(batch_sim[i][neg_indices])
                else:
                    # No in-batch negatives for this anchor
                    neg_sim_list.append(torch.tensor([], device=device))
            
            # Pad to uniform length
            if any(len(n) > 0 for n in neg_sim_list):
                max_neg = max(len(n) for n in neg_sim_list)
                neg_sim_padded = []
                for neg_sim_i in neg_sim_list:
                    if len(neg_sim_i) == 0:
                        # No negatives - use very small similarity
                        neg_sim_i = torch.full((max_neg,), -1e9, device=device)
                    elif len(neg_sim_i) < max_neg:
                        pad_size = max_neg - len(neg_sim_i)
                        padding = torch.full((pad_size,), -1e9, device=device)
                        neg_sim_i = torch.cat([neg_sim_i, padding])
                    neg_sim_padded.append(neg_sim_i)
                neg_sim = torch.stack(neg_sim_padded)  # [B, max_neg]
            else:
                neg_sim = None
        else:
            # No negatives available at all
            neg_sim = None
        
        # Compute contrastive loss
        num_pos = positive_embeds.shape[1]
        losses = []
        
        for i in range(num_pos):
            numerator = pos_sim[:, i]  # [B]
            
            if neg_sim is not None:
                # Standard contrastive loss with negatives
                all_sim = torch.cat([pos_sim, neg_sim], dim=1)
                denominator = torch.logsumexp(all_sim, dim=1)
            else:
                # No negatives - normalize over positives only
                # This encourages anchor to be most similar to its positives
                denominator = torch.logsumexp(pos_sim, dim=1)
            
            loss = -numerator + denominator
            losses.append(loss)
        
        total_loss = torch.stack(losses).mean()
        return total_loss


def get_decoder_embedding(model, input_ids, attention_mask):
    """
    Get sentence embedding from T5 decoder's first-step hidden state.
    This uses the exact representation the model uses for classification,
    ensuring no disruption to pre-trained knowledge.
    
    Args:
        model: T5 model
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
    Returns:
        embedding: [batch_size, hidden_dim]
    """
    # Encode input
    encoder_outputs = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
    
    # Prepare decoder start token
    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_input_ids = torch.full(
        (input_ids.shape[0], 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=input_ids.device
    )
    
    # Get decoder's first-step hidden state
    decoder_outputs = model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=attention_mask,
        return_dict=True
    )
    
    # Return first token's hidden state (sentence representation)
    return decoder_outputs.last_hidden_state[:, 0, :]  # [B, D]


def get_encoder_first_token(hidden_states):
    """
    Alternative: Use encoder's first token as sentence embedding.
    Simpler but less aligned with classification task than decoder.
    
    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]
    Returns:
        embedding: [batch_size, hidden_dim]
    """
    return hidden_states[:, 0, :]  # [B, D]


class ContrastiveDataset(Dataset):
    """
    Dataset that loads anchor + positives + negatives for contrastive learning.
    """
    def __init__(
        self, 
        data_args: DataArguments, 
        tokenizer: transformers.PreTrainedTokenizer, 
        split="train"
    ):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.split = split
        self.dataset_path = data_args.data_path

        self.use_contrastive = data_args.use_contrastive and split == "train"

        self.filter_error_types = data_args.filter_error_types 
        
        # Load dataset
        self.data = self.load_dataset(split, data_args)
        
        logging.info(f"Loaded {len(self.data)} examples for {split}")
        if self.use_contrastive:
            logging.info(f"Contrastive learning enabled: sampling {data_args.num_positives} pos, {data_args.num_negatives} neg")
    
    def load_dataset(self, split, data_args):
        features = Features(
            {
                "question": Value("string"),  # 字符串字段
                "claim": Value("string"),  # 字符串字段
                "claim_raw_string": Value("string"),  # 字符串字段
                "response": Value("string"),  # 字符串字段
                "references": datasets.Sequence(Value("string")),  # 字符串字段
                "citation_links": datasets.Sequence(Value("string")),  # 字符串字段
                "webpage_references": datasets.Sequence(Value("string")),  # 字符串字段
                "attribution_label": Value("string"),  # 字符串字段
                "src_dataset": Value("string"),  # 字符串字段
                "id": Value("string"),  # 字符串字段
            }
        )
        # Load the dataset
        data_path=os.environ['WORK']+"/AttributionBench"
        data = datasets.load_from_disk(data_path)
        if split in ["stanford_dev", "attributedqa_dev", "hagrid_dev", "expertqa_dev"]:
            dataset=data["dev"] 
        elif split == "train":
            # Load augmented dataset with positives/negatives
            # Assuming format: {"anchor": {...}, "positives": [...], "negatives": [...]}
            data_path=os.environ['WORK']+"/"+ data_args.dataset_version #"/AttributionBench"
            data = datasets.load_from_disk(data_path)
            dataset=data[split]
        else:
            data_path=os.environ['WORK']+"/AttributionBench"
            data = datasets.load_from_disk(data_path)

            dataset=data[split] 
        return dataset

    
    # def format_example(self, example):
    #     """Format a single example into input text."""
    #     query = example.get("question", "")
    #     if query in ["nan", "", None]:
    #         query = ""
        
    #     answer = example.get("claim", "")
    #     if answer in ["nan", "", None]:
    #         answer = ""
        
    #     documents = example.get("references", [])
    #     documents_concatenation = "\n\n\n".join(documents)
        
    #     input_template = "premise: {} hypothesis: {}"
        
    #     if query:
    #         input_text = input_template.format(documents_concatenation, " ".join([query, answer]))
    #     else:
    #         input_text = input_template.format(documents_concatenation, answer)
        
    #     return input_text


    def process_function(self, example):
        def format_prompt(
            example,
            have_question=False,
            have_response=False,
            prompt_name=self.data_args.template,
        ):
            query = (
                example["question"]
                if example["question"] and example["question"] not in ["nan", "", None]
                else ""
            )
            answer = (
                example["claim"]
                if example["claim"] and example["claim"] not in ["nan", "", None]
                else ""
            )
            response = (
                example["response"]
                if example["response"] and example["response"] not in ["nan", "", None]
                else ""
            )
            if "references" in example.keys() and len(example["references"]):
                documents_concatenation = "\n\n\n".join(example["references"])
            else:
                print("Empty references",example)

            if have_question and have_response:
                input_template = "### Input:\nQuestion: {}\n\nClaim: {}\n\nResponse: {}\n\nReference: {}\n\n### Output:"
                input = input_template.format(
                    query, answer, response, documents_concatenation
                )
            elif have_question and not have_response:
                input_template = "premise: {} hypothesis: {}"
                input = input_template.format(documents_concatenation, " ".join(query, answer))
                # input_template = "### Input:\nQuestion: {}\n\nClaim: {}\n\nReference: {}\n\n### Output:"
                # input = input_template.format(query, answer, documents_concatenation)
            elif not have_question and have_response:
                input_template = "### Input:\nClaim: {}\n\nResponse: {}\n\nReference: {}\n\n### Output:"
                input = input_template.format(answer, response, documents_concatenation)
            else:
                input_template = "premise: {} hypothesis: {}"
                input = input_template.format(documents_concatenation, answer)

            instructions = json.load(open(self.data_args.template_path))
            # formatted_prompt = "{}{}".format(instructions[prompt_name]["llama2"], input)
            formatted_prompt = input

            return formatted_prompt

        if "q_c_e_r" in self.data_args.template:
            have_question = True
            have_response = True
        elif "q_c_e" in self.data_args.template:
            have_question = True
            have_response = False
        elif "c_e_r" in self.data_args.template:
            have_question = False
            have_response = True
        else:
            have_question = False
            have_response = False

        source = format_prompt(
            example,
            have_question=have_question,
            have_response=have_response,
            prompt_name=self.data_args.template,
        )
        return source
    
    def tokenize_example(self, text, is_target=False):
        """Tokenize text."""
        if is_target:
            token_ids = self.tokenizer(
                text_target=text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids[0]
            token_ids = torch.where(token_ids == self.tokenizer.pad_token_id, -100, token_ids)
        else:
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            token_ids = encoding.input_ids[0]
            attention_mask = encoding.attention_mask[0]
            return token_ids, attention_mask
        
        return token_ids
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        if not self.use_contrastive:
            # Regular mode: just return anchor
            anchor_text = self.process_function(example)
            input_ids, attention_mask = self.tokenize_example(anchor_text)
            
            label = "1" if str(example.get('attribution_label', '')) == "attributable" else "0"
            labels = self.tokenize_example(label, is_target=True)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        # Check if this is an anchor-only example (no positives/negatives)

        has_positives = "positives" in example and len(example["positives"]) > 0
        has_negatives = "negatives" in example and len(example["negatives"]) > 0
        is_anchor_only = not has_positives and not has_negatives
        
        # Handle different example types
        if is_anchor_only:
            # Anchor-only: return with empty pos/neg for classification-only training
            anchor = example['anchor'] if "anchor" in example else example
            anchor_text = self.process_function(anchor)
            input_ids, attention_mask = self.tokenize_example(anchor_text)
            
            label = "1" if str(anchor.get('attribution_label', '')) == "attributable" else "0"
            labels = self.tokenize_example(label, is_target=True)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pos_input_ids": [],  # Empty positives
                "pos_attention_masks": [],
                "neg_input_ids": [],  # Empty negatives
                "neg_attention_masks": [],
                "is_anchor_only": True  # Flag for trainer
            }
        
        # Contrastive mode: return anchor + sampled positives + negatives
        anchor = example["anchor"] if "anchor" in example else example
        positives = example["positives"]
        negatives = example["negatives"]
        

        if self.filter_error_types: 
            print("Filtering Error types and leaving only", T5_TYPES)
            filtered_neg=[]
            filtered_pos=[]
            for n in negatives:
                if n["error_type"] in T5_TYPES:
                    filtered_neg.append(n)
            for p in positives:
                if p["error_type"] in T5_TYPES:
                    filtered_pos.append(p)
            positives=filtered_pos
            print("Filtering Positives", positives)
            negatives=filtered_neg
            print("Filtering negatives", negatives)

        # Sample subset
        num_pos = min(self.data_args.num_positives, len(positives))
        num_neg = min(self.data_args.num_negatives, len(negatives))
        
        sampled_positives = random.sample(positives, num_pos) if len(positives) > 0 else []
        sampled_negatives = random.sample(negatives, num_neg) if len(negatives) > 0 else []
        
        # Format and tokenize anchor
        anchor_text = self.process_function(anchor)
        anchor_input_ids, anchor_attention_mask = self.tokenize_example(anchor_text)
        
        anchor_label = "1" if str(anchor.get('attribution_label', '')) == "attributable" else "0" 
        anchor_labels = self.tokenize_example(anchor_label, is_target=True)
        
        # Format and tokenize positives
        pos_input_ids = []
        pos_attention_masks = []
        for pos in sampled_positives:
            pos_text = self.process_function(pos)
            pos_ids, pos_mask = self.tokenize_example(pos_text)
            pos_input_ids.append(pos_ids)
            pos_attention_masks.append(pos_mask)
        
        # Format and tokenize negatives
        neg_input_ids = []
        neg_attention_masks = []
        for neg in sampled_negatives:
            neg_text = self.process_function(neg)
            neg_ids, neg_mask = self.tokenize_example(neg_text)
            neg_input_ids.append(neg_ids)
            neg_attention_masks.append(neg_mask)
        
        return {
            "input_ids": anchor_input_ids,
            "attention_mask": anchor_attention_mask,
            "labels": anchor_labels,
            "pos_input_ids": pos_input_ids,
            "pos_attention_masks": pos_attention_masks,
            "neg_input_ids": neg_input_ids,
            "neg_attention_masks": neg_attention_masks,
            "is_anchor_only": False
        }

@dataclass
class ContrastiveDataCollator:
    """Collator for contrastive learning batches. Handles variable-length pos/neg lists."""
    tokenizer: transformers.PreTrainedTokenizer
    use_contrastive: bool = True
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad anchors
        anchor_input_ids = [inst["input_ids"] for inst in instances]
        anchor_attention_masks = [inst["attention_mask"] for inst in instances]
        anchor_labels = [inst["labels"] for inst in instances]
        
        anchor_input_ids = torch.nn.utils.rnn.pad_sequence(
            anchor_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        anchor_attention_masks = torch.nn.utils.rnn.pad_sequence(
            anchor_attention_masks, batch_first=True, padding_value=0
        )
        anchor_labels = torch.nn.utils.rnn.pad_sequence(
            anchor_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # If not using contrastive or no positives/negatives in batch, return simple batch
        if not self.use_contrastive:
            return {
                "input_ids": anchor_input_ids,
                "attention_mask": anchor_attention_masks,
                "labels": anchor_labels
            }
        
        # Check if any instance has contrastive data
        has_contrastive = any(
            len(inst.get("pos_input_ids", [])) > 0 or len(inst.get("neg_input_ids", [])) > 0
            for inst in instances
        )
        
        if not has_contrastive:
            # All anchor-only: return classification-only batch
            return {
                "input_ids": anchor_input_ids,
                "attention_mask": anchor_attention_masks,
                "labels": anchor_labels
            }
        
        # Pad positives
        all_pos_input_ids = []
        all_pos_attention_masks = []
        pos_counts = []
        
        for inst in instances:
            pos_ids = inst.get("pos_input_ids", [])
            pos_masks = inst.get("pos_attention_masks", [])
            pos_counts.append(len(pos_ids))
            all_pos_input_ids.extend(pos_ids)
            all_pos_attention_masks.extend(pos_masks)
        
        max_pos = max(pos_counts) if pos_counts else 0
        
        if max_pos > 0:
            # Pad sequences
            all_pos_input_ids = torch.nn.utils.rnn.pad_sequence(
                all_pos_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            all_pos_attention_masks = torch.nn.utils.rnn.pad_sequence(
                all_pos_attention_masks, batch_first=True, padding_value=0
            )
            
            # Reshape to [batch, max_pos, seq_len]
            pos_input_ids_batched = []
            pos_attention_masks_batched = []
            idx = 0
            for count in pos_counts:
                if count == 0:
                    # Dummy positives for anchor-only examples
                    dummy_ids = torch.full(
                        (max_pos, all_pos_input_ids.shape[1]),
                        self.tokenizer.pad_token_id,
                        dtype=torch.long
                    )
                    dummy_masks = torch.zeros((max_pos, all_pos_attention_masks.shape[1]), dtype=torch.long)
                    pos_input_ids_batched.append(dummy_ids)
                    pos_attention_masks_batched.append(dummy_masks)
                else:
                    batch_pos_ids = all_pos_input_ids[idx:idx+count]
                    batch_pos_masks = all_pos_attention_masks[idx:idx+count]
                    
                    if count < max_pos:
                        # Pad to max_pos
                        pad_amount = max_pos - count
                        pad_ids = torch.full((pad_amount, batch_pos_ids.shape[1]), 
                                           self.tokenizer.pad_token_id, dtype=torch.long)
                        pad_masks = torch.zeros((pad_amount, batch_pos_masks.shape[1]), dtype=torch.long)
                        batch_pos_ids = torch.cat([batch_pos_ids, pad_ids], dim=0)
                        batch_pos_masks = torch.cat([batch_pos_masks, pad_masks], dim=0)
                    
                    pos_input_ids_batched.append(batch_pos_ids)
                    pos_attention_masks_batched.append(batch_pos_masks)
                    idx += count
            
            pos_input_ids = torch.stack(pos_input_ids_batched)
            pos_attention_masks = torch.stack(pos_attention_masks_batched)
        else:
            pos_input_ids = None
            pos_attention_masks = None
        
        # Pad negatives
        all_neg_input_ids = []
        all_neg_attention_masks = []
        neg_counts = []
        
        for inst in instances:
            neg_ids = inst.get("neg_input_ids", [])
            neg_masks = inst.get("neg_attention_masks", [])
            neg_counts.append(len(neg_ids))
            all_neg_input_ids.extend(neg_ids)
            all_neg_attention_masks.extend(neg_masks)
        
        max_neg = max(neg_counts) if neg_counts else 0
        
        if max_neg > 0:
            all_neg_input_ids = torch.nn.utils.rnn.pad_sequence(
                all_neg_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            all_neg_attention_masks = torch.nn.utils.rnn.pad_sequence(
                all_neg_attention_masks, batch_first=True, padding_value=0
            )
            
            neg_input_ids_batched = []
            neg_attention_masks_batched = []
            idx = 0
            for count in neg_counts:
                if count == 0:
                    dummy_ids = torch.full(
                        (max_neg, all_neg_input_ids.shape[1]),
                        self.tokenizer.pad_token_id,
                        dtype=torch.long
                    )
                    dummy_masks = torch.zeros((max_neg, all_neg_attention_masks.shape[1]), dtype=torch.long)
                    neg_input_ids_batched.append(dummy_ids)
                    neg_attention_masks_batched.append(dummy_masks)
                else:
                    batch_neg_ids = all_neg_input_ids[idx:idx+count]
                    batch_neg_masks = all_neg_attention_masks[idx:idx+count]
                    
                    if count < max_neg:
                        pad_amount = max_neg - count
                        pad_ids = torch.full((pad_amount, batch_neg_ids.shape[1]),
                                           self.tokenizer.pad_token_id, dtype=torch.long)
                        pad_masks = torch.zeros((pad_amount, batch_neg_masks.shape[1]), dtype=torch.long)
                        batch_neg_ids = torch.cat([batch_neg_ids, pad_ids], dim=0)
                        batch_neg_masks = torch.cat([batch_neg_masks, pad_masks], dim=0)
                    
                    neg_input_ids_batched.append(batch_neg_ids)
                    neg_attention_masks_batched.append(batch_neg_masks)
                    idx += count
            
            neg_input_ids = torch.stack(neg_input_ids_batched)
            neg_attention_masks = torch.stack(neg_attention_masks_batched)
        else:
            neg_input_ids = None
            neg_attention_masks = None
        
        return {
            "input_ids": anchor_input_ids,
            "attention_mask": anchor_attention_masks,
            "labels": anchor_labels,
            "pos_input_ids": pos_input_ids,
            "pos_attention_masks": pos_attention_masks,
            "neg_input_ids": neg_input_ids,
            "neg_attention_masks": neg_attention_masks,
        }



# @dataclass
# class ContrastiveDataCollator:
#     """Collator for contrastive learning batches."""
#     tokenizer: transformers.PreTrainedTokenizer
#     use_contrastive: bool = True
    
#     def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
#         if len(instances):
#             if "pos_input_ids" not in instances[0].keys():
#                 contrastive_data= False
#             else:
#                 contrastive_data= True
#         if not self.use_contrastive or  not contrastive_data:
#             # Regular collation
#             input_ids = [inst["input_ids"] for inst in instances]
#             attention_masks = [inst["attention_mask"] for inst in instances]
#             labels = [inst["labels"] for inst in instances]
            
#             input_ids = torch.nn.utils.rnn.pad_sequence(
#                 input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#             )
#             attention_masks = torch.nn.utils.rnn.pad_sequence(
#                 attention_masks, batch_first=True, padding_value=0
#             )
#             labels = torch.nn.utils.rnn.pad_sequence(
#                 labels, batch_first=True, padding_value=IGNORE_INDEX
#             )
            
#             return {
#                 "input_ids": input_ids,
#                 "attention_mask": attention_masks,
#                 "labels": labels
#             }
        
#         # Contrastive collation
#         batch_size = len(instances)
        
#         # Anchors
#         anchor_input_ids = [inst["input_ids"] for inst in instances]
#         anchor_attention_masks = [inst["attention_mask"] for inst in instances]
#         anchor_labels = [inst["labels"] for inst in instances]
        
#         anchor_input_ids = torch.nn.utils.rnn.pad_sequence(
#             anchor_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         anchor_attention_masks = torch.nn.utils.rnn.pad_sequence(
#             anchor_attention_masks, batch_first=True, padding_value=0
#         )
#         anchor_labels = torch.nn.utils.rnn.pad_sequence(
#             anchor_labels, batch_first=True, padding_value=IGNORE_INDEX
#         )
        
#         # Positives: flatten, pad, then reshape
#         all_pos_input_ids = []
#         all_pos_attention_masks = []
#         pos_counts = []
        
#         for inst in instances:
#             pos_ids = inst["pos_input_ids"]
#             pos_masks = inst["pos_attention_masks"]
#             pos_counts.append(len(pos_ids))
#             all_pos_input_ids.extend(pos_ids)
#             all_pos_attention_masks.extend(pos_masks)
        
#         max_pos = max(pos_counts) if pos_counts else 0
        
#         if len(all_pos_input_ids) > 0:
#             all_pos_input_ids = torch.nn.utils.rnn.pad_sequence(
#                 all_pos_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#             )
#             all_pos_attention_masks = torch.nn.utils.rnn.pad_sequence(
#                 all_pos_attention_masks, batch_first=True, padding_value=0
#             )
            
#             # Pad to max_pos per batch item
#             pos_input_ids_batched = []
#             pos_attention_masks_batched = []
#             idx = 0
#             for count in pos_counts:
#                 batch_pos_ids = all_pos_input_ids[idx:idx+count]
#                 batch_pos_masks = all_pos_attention_masks[idx:idx+count]
                
#                 # Pad to max_pos
#                 if count < max_pos:
#                     pad_amount = max_pos - count
#                     pad_ids = torch.full(
#                         (pad_amount, batch_pos_ids.shape[1]), 
#                         self.tokenizer.pad_token_id, 
#                         dtype=batch_pos_ids.dtype
#                     )
#                     pad_masks = torch.zeros(
#                         (pad_amount, batch_pos_masks.shape[1]), 
#                         dtype=batch_pos_masks.dtype
#                     )
#                     batch_pos_ids = torch.cat([batch_pos_ids, pad_ids], dim=0)
#                     batch_pos_masks = torch.cat([batch_pos_masks, pad_masks], dim=0)
                
#                 pos_input_ids_batched.append(batch_pos_ids)
#                 pos_attention_masks_batched.append(batch_pos_masks)
#                 idx += count
            
#             pos_input_ids = torch.stack(pos_input_ids_batched)  # [B, max_pos, seq_len]
#             pos_attention_masks = torch.stack(pos_attention_masks_batched)
#         else:
#             pos_input_ids = None
#             pos_attention_masks = None
        
#         # Negatives: same process
#         all_neg_input_ids = []
#         all_neg_attention_masks = []
#         neg_counts = []
        
#         for inst in instances:
#             neg_ids = inst["neg_input_ids"]
#             neg_masks = inst["neg_attention_masks"]
#             neg_counts.append(len(neg_ids))
#             all_neg_input_ids.extend(neg_ids)
#             all_neg_attention_masks.extend(neg_masks)
        
#         max_neg = max(neg_counts) if neg_counts else 0
        
#         if len(all_neg_input_ids) > 0:
#             all_neg_input_ids = torch.nn.utils.rnn.pad_sequence(
#                 all_neg_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#             )
#             all_neg_attention_masks = torch.nn.utils.rnn.pad_sequence(
#                 all_neg_attention_masks, batch_first=True, padding_value=0
#             )
            
#             neg_input_ids_batched = []
#             neg_attention_masks_batched = []
#             idx = 0
#             for count in neg_counts:
#                 batch_neg_ids = all_neg_input_ids[idx:idx+count]
#                 batch_neg_masks = all_neg_attention_masks[idx:idx+count]
                
#                 if count < max_neg:
#                     pad_amount = max_neg - count
#                     pad_ids = torch.full(
#                         (pad_amount, batch_neg_ids.shape[1]), 
#                         self.tokenizer.pad_token_id, 
#                         dtype=batch_neg_ids.dtype
#                     )
#                     pad_masks = torch.zeros(
#                         (pad_amount, batch_neg_masks.shape[1]), 
#                         dtype=batch_neg_masks.dtype
#                     )
#                     batch_neg_ids = torch.cat([batch_neg_ids, pad_ids], dim=0)
#                     batch_neg_masks = torch.cat([batch_neg_masks, pad_masks], dim=0)
                
#                 neg_input_ids_batched.append(batch_neg_ids)
#                 neg_attention_masks_batched.append(batch_neg_masks)
#                 idx += count
            
#             neg_input_ids = torch.stack(neg_input_ids_batched)  # [B, max_neg, seq_len]
#             neg_attention_masks = torch.stack(neg_attention_masks_batched)
#         else:
#             neg_input_ids = None
#             neg_attention_masks = None
        
#         return {
#             "input_ids": anchor_input_ids,
#             "attention_mask": anchor_attention_masks,
#             "labels": anchor_labels,
#             "pos_input_ids": pos_input_ids,
#             "pos_attention_masks": pos_attention_masks,
#             "neg_input_ids": neg_input_ids,
#             "neg_attention_masks": neg_attention_masks,
#         }


# class ContrastiveTrainer(Seq2SeqTrainer):
#     """
#     Custom trainer that combines classification loss with contrastive loss.
#     Uses decoder's hidden state for contrastive learning to preserve pre-trained knowledge.
#     """
#     def __init__(self, *args, contrastive_loss_fn=None, contrastive_weight=0.5, 
#                  classification_weight=1.0, use_decoder_embedding=True, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.contrastive_loss_fn = contrastive_loss_fn
#         self.contrastive_weight = contrastive_weight
#         self.classification_weight = classification_weight
#         self.use_decoder_embedding = use_decoder_embedding
    
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         """
#         Compute combined loss: classification + contrastive.
#         """
#         # Extract contrastive inputs
#         pos_input_ids = inputs.pop("pos_input_ids", None)
#         pos_attention_masks = inputs.pop("pos_attention_masks", None)
#         neg_input_ids = inputs.pop("neg_input_ids", None)
#         neg_attention_masks = inputs.pop("neg_attention_masks", None)
        
#         # Regular forward pass for classification
#         outputs = model(**inputs)
#         classification_loss = outputs.loss
        
#         # If no contrastive data, return only classification loss
#         if pos_input_ids is None or neg_input_ids is None:
#             return (classification_loss, outputs) if return_outputs else classification_loss
        
#         # Contrastive learning using decoder embeddings (or encoder first token)
#         if self.use_decoder_embedding:
#             # Get decoder-based embeddings for anchor
#             anchor_embeds = get_decoder_embedding(
#                 model, 
#                 inputs["input_ids"], 
#                 inputs["attention_mask"]
#             )  # [B, D]
#         else:
#             # Alternative: use encoder's first token
#             encoder_outputs = model.encoder(
#                 input_ids=inputs["input_ids"],
#                 attention_mask=inputs["attention_mask"],
#                 return_dict=True
#             )
#             anchor_embeds = get_encoder_first_token(encoder_outputs.last_hidden_state)  # [B, D]
        
#         # Get embeddings for positives
#         batch_size, num_pos, seq_len = pos_input_ids.shape
#         pos_input_ids_flat = pos_input_ids.view(-1, seq_len)  # [B*N_pos, S]
#         pos_attention_masks_flat = pos_attention_masks.view(-1, seq_len)
        
#         if self.use_decoder_embedding:
#             pos_embeds_flat = get_decoder_embedding(
#                 model,
#                 pos_input_ids_flat,
#                 pos_attention_masks_flat
#             )  # [B*N_pos, D]
#         else:
#             pos_encoder_outputs = model.encoder(
#                 input_ids=pos_input_ids_flat,
#                 attention_mask=pos_attention_masks_flat,
#                 return_dict=True
#             )
#             pos_embeds_flat = get_encoder_first_token(pos_encoder_outputs.last_hidden_state)
        
#         pos_embeds = pos_embeds_flat.view(batch_size, num_pos, -1)  # [B, N_pos, D]
        
#         # Get embeddings for negatives
#         batch_size, num_neg, seq_len = neg_input_ids.shape
#         neg_input_ids_flat = neg_input_ids.view(-1, seq_len)
#         neg_attention_masks_flat = neg_attention_masks.view(-1, seq_len)
        
#         if self.use_decoder_embedding:
#             neg_embeds_flat = get_decoder_embedding(
#                 model,
#                 neg_input_ids_flat,
#                 neg_attention_masks_flat
#             )  # [B*N_neg, D]
#         else:
#             neg_encoder_outputs = model.encoder(
#                 input_ids=neg_input_ids_flat,
#                 attention_mask=neg_attention_masks_flat,
#                 return_dict=True
#             )
#             neg_embeds_flat = get_encoder_first_token(neg_encoder_outputs.last_hidden_state)
        
#         neg_embeds = neg_embeds_flat.view(batch_size, num_neg, -1)  # [B, N_neg, D]
        
#         # Compute contrastive loss
#         contrastive_loss = self.contrastive_loss_fn(anchor_embeds, pos_embeds, neg_embeds)
        
#         # Combined loss
#         total_loss = (
#             self.classification_weight * classification_loss + 
#             self.contrastive_weight * contrastive_loss
#         )
        
#         # Log individual losses
#         if self.state.global_step % 10 == 0:
#             logging.info(
#                 f"Step {self.state.global_step}: "
#                 f"Total={total_loss:.4f}, "
#                 f"Classification={classification_loss:.4f}, "
#                 f"Contrastive={contrastive_loss:.4f}"
#             )
        
#         return (total_loss, outputs) if return_outputs else total_loss


class ContrastiveTrainer(Seq2SeqTrainer):
    """
    Custom trainer that combines classification loss with contrastive loss.
    Uses decoder's hidden state for contrastive learning to preserve pre-trained knowledge.
    """
    def __init__(self, *args, contrastive_loss_fn=None, contrastive_weight=0.5, 
                 classification_weight=1.0, use_decoder_embedding=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss_fn = contrastive_loss_fn
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        self.use_decoder_embedding = use_decoder_embedding
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined loss: classification + contrastive.
        """
        # Extract contrastive inputs
        pos_input_ids = inputs.pop("pos_input_ids", None)
        pos_attention_masks = inputs.pop("pos_attention_masks", None)
        neg_input_ids = inputs.pop("neg_input_ids", None)
        neg_attention_masks = inputs.pop("neg_attention_masks", None)
        
        # Regular forward pass for classification
        outputs = model(**inputs)
        classification_loss = outputs.loss
        
        # If no contrastive data (no positives), return only classification loss
        if pos_input_ids is None or pos_input_ids.numel() == 0 or pos_input_ids.shape[1] == 0:
            return (classification_loss, outputs) if return_outputs else classification_loss
        
        # Get anchor labels for in-batch negative mining
        # Decode labels to get 0/1 values
        labels_decoded = inputs["labels"].clone()
        labels_decoded[labels_decoded == -100] = self.tokenizer.pad_token_id
        labels_text = self.tokenizer.batch_decode(labels_decoded, skip_special_tokens=True)
        anchor_labels = torch.tensor([int(l.strip() == "1") for l in labels_text], 
                                    device=inputs["labels"].device)
        
        # Contrastive learning using decoder embeddings (or encoder first token)
        if self.use_decoder_embedding:
            # Get decoder-based embeddings for anchor
            anchor_embeds = get_decoder_embedding(
                model, 
                inputs["input_ids"], 
                inputs["attention_mask"]
            )  # [B, D]
        else:
            # Alternative: use encoder's first token
            encoder_outputs = model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True
            )
            anchor_embeds = get_encoder_first_token(encoder_outputs.last_hidden_state)  # [B, D]
        
        # Get embeddings for positives
        batch_size, num_pos, seq_len = pos_input_ids.shape
        pos_input_ids_flat = pos_input_ids.view(-1, seq_len)  # [B*N_pos, S]
        pos_attention_masks_flat = pos_attention_masks.view(-1, seq_len)
        
        if self.use_decoder_embedding:
            pos_embeds_flat = get_decoder_embedding(
                model,
                pos_input_ids_flat,
                pos_attention_masks_flat
            )  # [B*N_pos, D]
        else:
            pos_encoder_outputs = model.encoder(
                input_ids=pos_input_ids_flat,
                attention_mask=pos_attention_masks_flat,
                return_dict=True
            )
            pos_embeds_flat = get_encoder_first_token(pos_encoder_outputs.last_hidden_state)
        
        pos_embeds = pos_embeds_flat.view(batch_size, num_pos, -1)  # [B, N_pos, D]
        
        # Get embeddings for negatives (if available)
        neg_embeds = None
        if neg_input_ids is not None and neg_input_ids.numel() > 0 and neg_input_ids.shape[1] > 0:
            batch_size, num_neg, seq_len = neg_input_ids.shape
            neg_input_ids_flat = neg_input_ids.view(-1, seq_len)
            neg_attention_masks_flat = neg_attention_masks.view(-1, seq_len)
            
            if self.use_decoder_embedding:
                neg_embeds_flat = get_decoder_embedding(
                    model,
                    neg_input_ids_flat,
                    neg_attention_masks_flat
                )  # [B*N_neg, D]
            else:
                neg_encoder_outputs = model.encoder(
                    input_ids=neg_input_ids_flat,
                    attention_mask=neg_attention_masks_flat,
                    return_dict=True
                )
                neg_embeds_flat = get_encoder_first_token(neg_encoder_outputs.last_hidden_state)
            
            neg_embeds = neg_embeds_flat.view(batch_size, num_neg, -1)  # [B, N_neg, D]
        
        # Compute contrastive loss with in-batch negatives fallback
        contrastive_loss = self.contrastive_loss_fn(
            anchor_embeds=anchor_embeds,
            positive_embeds=pos_embeds,
            negative_embeds=neg_embeds,
            anchor_labels=anchor_labels,
            all_batch_embeds=anchor_embeds,  # Use all anchors for in-batch negatives
            all_batch_labels=anchor_labels
        )
        
        # Combined loss
        total_loss = (
            self.classification_weight * classification_loss + 
            self.contrastive_weight * contrastive_loss
        )
        
        # Log individual losses
        if self.state.global_step % 10 == 0:
            has_explicit_neg = neg_embeds is not None
            logging.info(
                f"Step {self.state.global_step}: "
                f"Total={total_loss:.4f}, "
                f"Classification={classification_loss:.4f}, "
                f"Contrastive={contrastive_loss:.4f} "
                f"(explicit_neg={has_explicit_neg})"
            )
        
        return (total_loss, outputs) if return_outputs else total_loss


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    print("Computing Metric")
    logits = (
        eval_preds.predictions[0]
        if isinstance(eval_preds.predictions, tuple)
        else eval_preds.predictions
    )
    max_length = 128
    logits = logits[:, :max_length, :]
    preds = np.argmax(logits, axis=-1)
    labels = eval_preds.label_ids
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = [int(p.startswith(l)) for p, l in zip(decoded_preds, decoded_labels)]
    return {"accuracy": sum(result) / len(result)}


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning with contrastive learning."""
    split_train = "train"
    split_eval = "dev"
    split_eval_ood = "test_ood"

    print("train preparation")
    train_dataset = ContrastiveDataset(
        tokenizer=tokenizer, data_args=data_args, split="train"
    )
    

    data_collator = ContrastiveDataCollator(
        tokenizer=tokenizer, 
        use_contrastive=data_args.use_contrastive
    )
    
    # Eval datasets use regular format (no contrastive)
    eval_data_args = copy.deepcopy(data_args)
    eval_data_args.use_contrastive = False
    
    print("dev preparation")
    eval_dataset = ContrastiveDataset(
        tokenizer=tokenizer, data_args=eval_data_args, split="dev"
    )
    
    eval_collator = ContrastiveDataCollator(
        tokenizer=tokenizer,
        use_contrastive=False
    )
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Suppress wandb
    training_args.report_to = "wandb" #[] 

    with open(data_args.template_path) as f:
        template = json.load(f)

    model = transformers.T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    
    global tokenizer
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # Initialize contrastive loss
    contrastive_loss_fn = SupervisedContrastiveLoss(
        temperature=training_args.contrastive_temperature
    )
    
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Use custom trainer
    trainer = ContrastiveTrainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        args=training_args,
        contrastive_loss_fn=contrastive_loss_fn,
        contrastive_weight=training_args.contrastive_weight,
        classification_weight=training_args.classification_weight,
        use_decoder_embedding=training_args.use_decoder_embedding,
        **data_module,
    )
    
    trainer.train()
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    print(f"Model Saved to : {training_args.output_dir}")


if __name__ == "__main__":
    train()
