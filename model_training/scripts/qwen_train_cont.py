import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from torch.utils.data import Dataset
import datasets
from datasets import Features, Value
import json
import os
import random
from peft import LoraConfig, get_peft_model, TaskType
from collections import defaultdict

os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
os.environ['WANDB_MODE'] = 'offline'

IGNORE_INDEX = -100
SHOW_BATCH_SIZE = 0

T5_TYPES = ["claim_erronous_change", "claim_numerical_mismatch", "modify_passage-add_relevant_to_claim",
            "claim_combine_facts", "claim_add_to_the_claim_contradicting_info", "modify_passage-add_contradiction",
            "modify_passage-add_conflicting_sources", "claim_infer_claim", "claim_over_infer_claim"]


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-4B")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dev_data_path: str = field(default=None, metadata={"help": "Path to the dev data."})
    dataset_version: str = field(default="v3.0", metadata={"help": "Dataset version"})
    error_type: str = field(default=None)
    template: str = field(default="base_c_e")
    template_path: str = field(default="src/train/template.json")
    use_contrastive: bool = field(default=True, metadata={"help": "Use contrastive learning"})
    num_positives: int = field(default=4, metadata={"help": "Number of positives to sample per anchor"})
    num_negatives: int = field(default=4, metadata={"help": "Number of negatives to sample per anchor"})
    include_anchor_only: bool = field(
        default=True,
        metadata={"help": "Include examples with no positives/negatives (anchor-only)"}
    )
    filter_error_types: bool = field(
        default=False,
        metadata={"help": "Filter error types only leaving select ones"}
    )


@dataclass
class ContrastiveTrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length"})
    contrastive_weight: float = field(default=0.5, metadata={"help": "Weight for contrastive loss"})
    classification_weight: float = field(default=1.0, metadata={"help": "Weight for classification loss"})
    contrastive_temperature: float = field(default=0.07, metadata={"help": "Temperature for contrastive loss"})
    use_in_batch_negatives: bool = field(
        default=True,
        metadata={"help": "Fall back to in-batch negatives when explicit pos/neg are missing"}
    )
    use_in_batch_positives: bool = field(
        default=False,
        metadata={"help": "Uses in batch positives when positives are missing"}
    )


class HardNegativeContrastiveLoss(nn.Module):
    """
    Contrastive loss with explicit hard positives/negatives, with per-anchor dispatch:

      pos + neg  -> SupCon over explicit pairs
      neg only   -> repulsion (minimize cosine sim to negatives);
                    if use_in_batch_negatives, upgrade to SupCon with in-batch same-label anchors
      pos only   -> SupCon with in-batch negatives (requires use_in_batch_negatives)
      neither    -> in-batch SupCon (if use_in_batch_negatives) or zero (classification fallback)
    """

    def __init__(self, temperature: float = 0.07, use_in_batch_negatives: bool = True, use_in_batch_positives:bool = False):
        super().__init__()
        self.temperature = temperature
        self.use_in_batch_negatives = use_in_batch_negatives
        self.use_in_batch_positives = use_in_batch_positives
        self.total_batch_count = 0
        self.degenerate_batch_count = 0
        print(f"HardNegativeContrastiveLoss: temperature={temperature}, "
              f"use_in_batch_negatives={use_in_batch_negatives}",
              f"use_in_batch_positives={use_in_batch_positives}")

    def forward(
        self,
        anchor_embeds: torch.Tensor,      # [B, D]
        pos_embeds: torch.Tensor,         # [P_total, D]  (may have shape [0, D])
        neg_embeds: torch.Tensor,         # [N_total, D]  (may have shape [0, D])
        pos_anchor_idx: torch.Tensor,     # [P_total]  long, anchor index per positive
        neg_anchor_idx: torch.Tensor,     # [N_total]  long, anchor index per negative
        anchor_labels: torch.Tensor,      # [B]        long, 0/1 classification labels
    ) -> torch.Tensor:
        self.total_batch_count += 1
        B = anchor_embeds.shape[0]
        device = anchor_embeds.device

        if B < 1:
            print(f"Batch size less than 1: B={B}, returning 0.0 contrastive loss")
            return torch.tensor(0.0, device=device, requires_grad=True)

        anchor_embeds = F.normalize(anchor_embeds, dim=-1)
        if pos_embeds.shape[0] > 0:
            pos_embeds = F.normalize(pos_embeds, dim=-1)
        if neg_embeds.shape[0] > 0:
            neg_embeds = F.normalize(neg_embeds, dim=-1)

        per_anchor_losses = []
        for i in range(B):
            loss_i = self._loss_for_anchor(
                i, anchor_embeds, pos_embeds, neg_embeds,
                pos_anchor_idx, neg_anchor_idx, anchor_labels,
            )
            if loss_i is not None:
                print("Computed Loss with explicit Pos and Neg")
                per_anchor_losses.append(loss_i)
        print("Computed per_anchor_losses", len(per_anchor_losses))
        if not per_anchor_losses:
            self.degenerate_batch_count += 1
            logging.warning(
                f"[ContrastiveLoss] Degenerate batch "
                f"({self.degenerate_batch_count}/{self.total_batch_count}): "
                f"no valid anchors — falling back to classification loss."
            )
            #return torch.tensor(0.0, device=device, requires_grad=True)
            return 0.0 * anchor_embeds.sum()


        return torch.stack(per_anchor_losses).mean()

    def _loss_for_anchor(
        self, i: int,
        anchor_embeds: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
        pos_anchor_idx: torch.Tensor,
        neg_anchor_idx: torch.Tensor,
        anchor_labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        a = anchor_embeds[i:i+1]  # [1, D]

        explicit_pos = pos_embeds[pos_anchor_idx == i] if pos_embeds.shape[0] > 0 else pos_embeds[:0]
        explicit_neg = neg_embeds[neg_anchor_idx == i] if neg_embeds.shape[0] > 0 else neg_embeds[:0]

        has_pos = explicit_pos.shape[0] > 0
        has_neg = explicit_neg.shape[0] > 0

        # Case 1: explicit pos + explicit neg -> full SupCon
        if has_pos and has_neg:
            print("Both Explicit Pos and Neg ")
            return self._supcon(a, explicit_pos, explicit_neg)

        # Case 2: explicit pos only -> SupCon with in-batch negatives
        if has_pos:
            print(" Pos only ")
            if self.use_in_batch_negatives:
                ib_neg = self._inbatch_diff_label(i, anchor_embeds, anchor_labels)
                if ib_neg.shape[0] > 0:
                    return self._supcon(a, explicit_pos, ib_neg)
            print("No inBatch Negatives ")
            return None

        # Case 3: explicit neg only
        if has_neg:
            print(" Neg only ")
            if self.use_in_batch_positives:
                ib_pos = self._inbatch_same_label(i, anchor_embeds, anchor_labels)
                if ib_pos.shape[0] > 0:
                    return self._supcon(a, ib_pos, explicit_neg)
            # Pure repulsion: minimize cosine similarity to hard negatives
            print("No in batch positives, using repulsion")
            sim = (a @ explicit_neg.T).squeeze(0)  # [n_i]
            return sim.mean()

        # Case 4: neither explicit pos nor neg
        if self.use_in_batch_negatives:
            print("No Neg or Pos, using inBatch ")
            ib_pos = self._inbatch_same_label(i, anchor_embeds, anchor_labels)
            ib_neg = self._inbatch_diff_label(i, anchor_embeds, anchor_labels)
            if ib_pos.shape[0] > 0 and ib_neg.shape[0] > 0:
                return self._supcon(a, ib_pos, ib_neg)

        return None

    def _supcon(self, a: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        t = self.temperature
        sim_pos = (a @ pos.T).squeeze(0) / t   # [p]
        sim_neg = (a @ neg.T).squeeze(0) / t   # [n]
        all_sim = torch.cat([sim_pos, sim_neg])  # [p+n]
        shift = all_sim.max().detach()
        log_denom = torch.log(torch.exp(all_sim - shift).sum() + 1e-8)
        log_probs = (sim_pos - shift) - log_denom  # [p]
        return -log_probs.mean()

    def _inbatch_same_label(self, i, anchor_embeds, anchor_labels):
        mask = anchor_labels == anchor_labels[i]
        mask[i] = False
        return anchor_embeds[mask]

    def _inbatch_diff_label(self, i, anchor_embeds, anchor_labels):
        mask = anchor_labels != anchor_labels[i]
        return anchor_embeds[mask]


def get_last_token_embedding(
    hidden_states: torch.Tensor,  # [N, L, D]
    attention_mask: torch.Tensor,  # [N, L]
) -> torch.Tensor:                 # [N, D]
    """
    Extract the hidden state at the last real (non-padding) token.

    For our format [prompt tokens][answer token][pad...], the last real token
    is the answer token — the position the model uses to make its classification
    decision, analogous to T5's decoder first-step hidden state.
    """
    N = hidden_states.shape[0]
    last_idx = attention_mask.sum(dim=1) - 1  # [N]
    return hidden_states[torch.arange(N, device=hidden_states.device), last_idx, :]


class ContrastiveDataset(Dataset):
    """
    Dataset that loads anchor + optional explicit positives/negatives for contrastive learning.
    Each example is formatted as a single causal LM sequence: [prompt tokens][answer token].
    """
    def __init__(self, data_args: DataArguments, tokenizer, split="train"):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.split = split
        self.use_contrastive = data_args.use_contrastive and split == "train"
        self.filter_error_types = data_args.filter_error_types
        self.data = self.load_dataset(split, data_args)
        logging.info(f"Loaded {len(self.data)} examples for {split}")

    def stratified_shuffle(self, data, batch_size):
        attributable = [item for item in data
                        if (item.get('anchor', item))['attribution_label'] == 'attributable']
        non_attributable = [item for item in data
                            if (item.get('anchor', item))['attribution_label'] != 'attributable']
        random.shuffle(attributable)
        random.shuffle(non_attributable)
        result = []
        min_len = min(len(attributable), len(non_attributable))
        for i in range(min_len):
            pair = [attributable[i], non_attributable[i]]
            random.shuffle(pair)
            result.extend(pair)
        remainder = attributable[min_len:] + non_attributable[min_len:]
        random.shuffle(remainder)
        result.extend(remainder)
        return result

    def load_dataset(self, split, data_args):
        data_path = os.environ['WORK'] + "/AttributionBench"
        data = datasets.load_from_disk(data_path)
        if split in ["stanford_dev", "attributedqa_dev", "hagrid_dev", "expertqa_dev"]:
            dataset = data["dev"]
        elif split == "train":
            data_path = os.environ['WORK'] + "/" + data_args.dataset_version
            data = datasets.load_from_disk(data_path)
            dataset = data[split]
            dataset = self.stratified_shuffle(dataset, 4)
            for ishuf in range(4):
                anchor = dataset[ishuf]['anchor'] if "anchor" in dataset[ishuf] else dataset[ishuf]
                print(anchor["attribution_label"])
        else:
            dataset = data[split]
        return dataset

    # Fixed instruction (space after period, consistent casing)
    _INSTRUCTION = (
        "You will be given a claim and a document. "
        "Determine whether the claim is 'GROUNDED' or 'NOT GROUNDED' based on the document. "
        "A 'GROUNDED' claim is fully supported by the information provided in the document. "
        "It should be directly verifiable from the document. "
        "Only return the classification as the answer: 1 for 'GROUNDED' or 0 for 'NOT GROUNDED' "
        "without any explanation."
    )

    def process_function(self, example):
        """
        Returns (prefix, document, suffix) so that tokenize_causal can truncate only
        the document portion, guaranteeing that task keywords (CLAIM:, CLASSIFICATION:)
        are always present regardless of document length.
        """
        claim = example.get("claim") or ""
        if claim in ["nan", ""]:
            claim = ""
        query = example.get("question") or ""
        if query in ["nan", ""]:
            query = ""
        response = example.get("response") or ""
        if response in ["nan", ""]:
            response = ""

        if example.get("references"):
            document = "\n\n\n".join(example["references"])
        else:
            print("Empty references", example)
            document = ""

        inst = self._INSTRUCTION
        #print("Instruction:", inst)
        if "q_c_e_r" in self.data_args.template:
            prefix = f"{inst}\n\nQUESTION: {query}\n\nCLAIM: {claim}\n\nRESPONSE: {response}\n\nDOCUMENT: "
        elif "q_c_e" in self.data_args.template:
            prefix = f"{inst}\n\nQUESTION: {query}\n\nCLAIM: {claim}\n\nDOCUMENT: "
        elif "c_e_r" in self.data_args.template:
            prefix = f"{inst}\n\nCLAIM: {claim}\n\nRESPONSE: {response}\n\nDOCUMENT: "
        else:
            prefix = f"{inst}\n\nCLAIM: {claim}\n\nDOCUMENT: "

        suffix = "\n\nCLASSIFICATION:"
        return prefix, document, suffix

    def tokenize_causal(self, prefix: str, document: str, suffix: str, label_str: str):
        """
        Tokenize [prefix][document][suffix][answer_token] as a single causal LM sequence,
        truncating ONLY the document if total length exceeds model_max_length.

        This guarantees that the instruction and CLASSIFICATION: keyword are always present,
        even when references are very long.
        """
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=True)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(label_str, add_special_tokens=False)

        doc_budget = self.tokenizer.model_max_length - len(prefix_ids) - len(suffix_ids) - len(answer_ids)
        doc_ids = self.tokenizer.encode(document, add_special_tokens=False) if document else []
        if doc_budget <= 0:
            doc_ids = []
        elif len(doc_ids) > doc_budget:
            doc_ids = doc_ids[:doc_budget]

        input_ids = prefix_ids + doc_ids + suffix_ids + answer_ids
        attention_mask = [1] * len(input_ids)
        prompt_len = len(prefix_ids) + len(doc_ids) + len(suffix_ids)
        labels = [IGNORE_INDEX] * prompt_len + answer_ids

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        anchor = example.get('anchor', example)

        prefix, document, suffix = self.process_function(anchor)
        label_str = "1" if str(anchor.get('attribution_label', '')) == "attributable" else "0"
        input_ids, attention_mask, labels = self.tokenize_causal(prefix, document, suffix, label_str)

        result = {
            "input_ids": input_ids,          # [L_a]
            "attention_mask": attention_mask, # [L_a]
            "labels": labels,                 # [L_a], -100 for prompt, answer_id at last pos
        }

        if self.use_contrastive:
            pos_input_ids_list, pos_attention_mask_list = [], []
            neg_input_ids_list, neg_attention_mask_list = [], []
            print("processing Positives and Negatives in Data module")

            for pos_ex in (example.get("positives") or [])[:self.data_args.num_positives]:
                p_pre, p_doc, p_suf = self.process_function(pos_ex)
                p_label = "1" if str(pos_ex.get('attribution_label', '')) == "attributable" else "0"
                ids, mask, _ = self.tokenize_causal(p_pre, p_doc, p_suf, p_label)
                pos_input_ids_list.append(ids)
                pos_attention_mask_list.append(mask)

            for neg_ex in (example.get("negatives") or [])[:self.data_args.num_negatives]:
                n_pre, n_doc, n_suf = self.process_function(neg_ex)
                n_label = "1" if str(neg_ex.get('attribution_label', '')) == "attributable" else "0"
                ids, mask, _ = self.tokenize_causal(n_pre, n_doc, n_suf, n_label)
                neg_input_ids_list.append(ids)
                neg_attention_mask_list.append(mask)

            print("Len Pos and negative in Data module:", len(pos_attention_mask_list),len(neg_input_ids_list))

            result["pos_input_ids"] = pos_input_ids_list        # List[Tensor[L_p_j]]
            result["pos_attention_masks"] = pos_attention_mask_list
            result["neg_input_ids"] = neg_input_ids_list        # List[Tensor[L_n_j]]
            result["neg_attention_masks"] = neg_attention_mask_list

        return result


@dataclass
class ContrastiveDataCollator:
    """
    Collator for causal LM contrastive batches.
    Pads anchors into [B, L_a] and flattens all pos/neg into [P_total, L_p] / [N_total, L_n]
    with anchor-index trackers pos_anchor_idx / neg_anchor_idx.
    Skips pos/neg processing entirely when no instance in the batch has those keys
    (eval batches), preventing unexpected keys from reaching the base model.
    """
    tokenizer: transformers.PreTrainedTokenizer
    use_contrastive: bool = True

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        anchor_input_ids = torch.nn.utils.rnn.pad_sequence(
            [inst["input_ids"] for inst in instances],
            batch_first=True, padding_value=self.tokenizer.pad_token_id
        )  # [B, L_a]
        anchor_attention_masks = torch.nn.utils.rnn.pad_sequence(
            [inst["attention_mask"] for inst in instances],
            batch_first=True, padding_value=0
        )  # [B, L_a]
        anchor_labels = torch.nn.utils.rnn.pad_sequence(
            [inst["labels"] for inst in instances],
            batch_first=True, padding_value=IGNORE_INDEX
        )  # [B, L_a]

        batch = {
            "input_ids": anchor_input_ids,
            "attention_mask": anchor_attention_masks,
            "labels": anchor_labels,
        }

        # Only process pos/neg when instances actually carry them (train, not eval)
        has_contrastive = self.use_contrastive and any("pos_input_ids" in inst for inst in instances)
        if not has_contrastive:
            print("!!!! X !!!!! Not using contrastive")
            return batch

        all_pos_ids, all_pos_masks, pos_anchor_idx = [], [], []
        all_neg_ids, all_neg_masks, neg_anchor_idx = [], [], []

        for batch_i, inst in enumerate(instances):
            for ids, mask in zip(inst.get("pos_input_ids", []), inst.get("pos_attention_masks", [])):
                all_pos_ids.append(ids)
                all_pos_masks.append(mask)
                pos_anchor_idx.append(batch_i)
            for ids, mask in zip(inst.get("neg_input_ids", []), inst.get("neg_attention_masks", [])):
                all_neg_ids.append(ids)
                all_neg_masks.append(mask)
                neg_anchor_idx.append(batch_i)

        if all_pos_ids:
            batch["pos_input_ids"] = torch.nn.utils.rnn.pad_sequence(
                all_pos_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )  # [P_total, L_p]
            batch["pos_attention_masks"] = torch.nn.utils.rnn.pad_sequence(
                all_pos_masks, batch_first=True, padding_value=0
            )
            batch["pos_anchor_idx"] = torch.tensor(pos_anchor_idx, dtype=torch.long)
        else:
            batch["pos_input_ids"] = torch.zeros(0, 1, dtype=torch.long)
            batch["pos_attention_masks"] = torch.zeros(0, 1, dtype=torch.long)
            batch["pos_anchor_idx"] = torch.zeros(0, dtype=torch.long)

        if all_neg_ids:
            batch["neg_input_ids"] = torch.nn.utils.rnn.pad_sequence(
                all_neg_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )  # [N_total, L_n]
            batch["neg_attention_masks"] = torch.nn.utils.rnn.pad_sequence(
                all_neg_masks, batch_first=True, padding_value=0
            )
            batch["neg_anchor_idx"] = torch.tensor(neg_anchor_idx, dtype=torch.long)
        else:
            batch["neg_input_ids"] = torch.zeros(0, 1, dtype=torch.long)
            batch["neg_attention_masks"] = torch.zeros(0, 1, dtype=torch.long)
            batch["neg_anchor_idx"] = torch.zeros(0, dtype=torch.long)

        return batch


_CONTRASTIVE_KEYS = frozenset([
    "pos_input_ids", "pos_attention_masks", "pos_anchor_idx",
    "neg_input_ids", "neg_attention_masks", "neg_anchor_idx",
])


class ContrastiveTrainer(Trainer):
    """
    Custom Trainer that combines causal LM classification loss with hard-negative contrastive loss.
    Embedding: last real token hidden state (answer token position) — equivalent to T5's
    decoder first-step, since it is the representation used to predict the answer.
    """
    def __init__(self, *args, contrastive_loss_fn=None, contrastive_weight=0.5,
                 classification_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss_fn = contrastive_loss_fn
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        self.custom_loss_tracker = {'classification': [], 'contrastive': []}

    def _get_embeddings(
        self,
        model,
        input_ids: torch.Tensor,      # [N, L]
        attention_mask: torch.Tensor,  # [N, L]
    ) -> torch.Tensor:                 # [N, D]
        """Run a forward pass and return the last-token embedding for each sequence."""
        if input_ids.shape[0] == 0:
            D = (model.module if hasattr(model, "module") else model).config.hidden_size
            return torch.zeros(0, D, device=input_ids.device)

        model_inner = model.module if hasattr(model, "module") else model
        out = model_inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return get_last_token_embedding(out.hidden_states[-1], attention_mask)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Strip contrastive-only keys before the eval forward pass."""
        inputs = {k: v for k, v in inputs.items() if k not in _CONTRASTIVE_KEYS}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop contrastive tensors — AutoModelForCausalLM does not accept them
        pos_input_ids = inputs.pop("pos_input_ids", None)
        pos_attention_masks = inputs.pop("pos_attention_masks", None)
        pos_anchor_idx = inputs.pop("pos_anchor_idx", None)
        neg_input_ids = inputs.pop("neg_input_ids", None)
        neg_attention_masks = inputs.pop("neg_attention_masks", None)
        neg_anchor_idx = inputs.pop("neg_anchor_idx", None)

        # Main forward pass on anchors
        outputs = model(**inputs, output_hidden_states=True)
        classification_loss = outputs.loss

        # Anchor embeddings: hidden state at the answer token position
        anchor_embeds = get_last_token_embedding(
            outputs.hidden_states[-1], inputs["attention_mask"]
        )  # [B, D]

        # Derive 0/1 anchor labels from the labels tensor for in-batch fallback
        labels_decoded = inputs["labels"].clone()
        _tok = getattr(self, 'processing_class', None) or self.tokenizer
        labels_decoded[labels_decoded == IGNORE_INDEX] = _tok.pad_token_id
        labels_text = _tok.batch_decode(labels_decoded, skip_special_tokens=True)

        # labels_decoded[labels_decoded == IGNORE_INDEX] = self.tokenizer.pad_token_id
        # labels_text = self.tokenizer.batch_decode(labels_decoded, skip_special_tokens=True)
        anchor_labels = torch.tensor(
            [int(t.strip() == "1") for t in labels_text],
            device=inputs["labels"].device
        )  # [B]

        contrastive_loss = torch.tensor(0.0, device=anchor_embeds.device)

        if self.contrastive_loss_fn is not None:
            device = anchor_embeds.device
            D = anchor_embeds.shape[-1]

            if pos_input_ids is not None and pos_input_ids.shape[0] > 0:
                pos_embeds = self._get_embeddings(
                    model, pos_input_ids.to(device), pos_attention_masks.to(device)
                )  # [P_total, D]
                pos_anchor_idx = pos_anchor_idx.to(device)
            else:
                pos_embeds = torch.zeros(0, D, device=device)
                pos_anchor_idx = torch.zeros(0, dtype=torch.long, device=device)

            if neg_input_ids is not None and neg_input_ids.shape[0] > 0:
                neg_embeds = self._get_embeddings(
                    model, neg_input_ids.to(device), neg_attention_masks.to(device)
                )  # [N_total, D]
                neg_anchor_idx = neg_anchor_idx.to(device)
            else:
                neg_embeds = torch.zeros(0, D, device=device)
                neg_anchor_idx = torch.zeros(0, dtype=torch.long, device=device)

            contrastive_loss = self.contrastive_loss_fn(
                anchor_embeds,   # [B, D]
                pos_embeds,      # [P_total, D]
                neg_embeds,      # [N_total, D]
                pos_anchor_idx,  # [P_total]
                neg_anchor_idx,  # [N_total]
                anchor_labels,   # [B]
            )

        total_loss = (
            self.classification_weight * classification_loss +
            self.contrastive_weight * contrastive_loss
        )

        if self.state.global_step % 10 == 0:
            print(
                f"Step {self.state.global_step}: "
                f"Total={total_loss:.4f}, "
                f"Classification={classification_loss:.4f}, "
                f"Contrastive={contrastive_loss:.4f}"
            )
            self.custom_loss_tracker['classification'].append(classification_loss.item())
            self.custom_loss_tracker['contrastive'].append(contrastive_loss.item())

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self.custom_loss_tracker['classification']:
            logs["class_loss"] = (sum(self.custom_loss_tracker['classification']) /
                                  len(self.custom_loss_tracker['classification']))
            logs["cont_loss"] = (sum(self.custom_loss_tracker['contrastive']) /
                                 len(self.custom_loss_tracker['contrastive']))
            self.custom_loss_tracker['classification'] = []
            self.custom_loss_tracker['contrastive'] = []
        super().log(logs, *args, **kwargs)


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compress [B, L, V] logits to [B, V] by extracting only the logit that predicts
    the answer token, saving memory during evaluation.

    In a causal LM, logits[b, i, :] predicts input_ids[b, i+1].
    The answer token is at position p (first non-(-100) label), so the predicting
    logit is at position p-1.
    """
    B = logits.shape[0]
    answer_pos = (labels != IGNORE_INDEX).long().argmax(dim=1)  # [B]
    pred_pos = (answer_pos - 1).clamp(min=0)                    # [B]
    return logits[torch.arange(B, device=logits.device), pred_pos, :]  # [B, V]


def compute_metrics(eval_preds):
    # predictions: [B, V] after preprocess_logits_for_metrics
    answer_logits = eval_preds.predictions
    labels = eval_preds.label_ids
    B = answer_logits.shape[0]
    answer_pos = (labels != IGNORE_INDEX).argmax(axis=1)  # [B]
    true_ids = labels[np.arange(B), answer_pos]            # [B]
    pred_ids = answer_logits.argmax(axis=-1)                # [B]
    return {"accuracy": float((pred_ids == true_ids).mean())}


def make_supervised_data_module(tokenizer, data_args) -> Dict:
    print("train preparation")
    train_dataset = ContrastiveDataset(tokenizer=tokenizer, data_args=data_args, split="train")

    data_collator = ContrastiveDataCollator(
        tokenizer=tokenizer,
        use_contrastive=data_args.use_contrastive,
    )

    eval_data_args = copy.deepcopy(data_args)
    eval_data_args.use_contrastive = False

    print("dev preparation")
    eval_dataset = ContrastiveDataset(tokenizer=tokenizer, data_args=eval_data_args, split="dev")

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, ContrastiveTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.report_to = "wandb"

    with open(data_args.template_path) as f:
        template = json.load(f)

    # Load Qwen3-4B in bfloat16 (recommended for Qwen3)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    # LoRA for decoder-only LM: target_modules use _proj suffix, task_type=CAUSAL_LM
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    #model.enable_input_require_grads()
    model.print_trainable_parameters()

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    # Qwen3 may not define a pad token; reuse eos_token (masked by attention_mask during training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    contrastive_loss_fn = HardNegativeContrastiveLoss(
        temperature=training_args.contrastive_temperature,
        use_in_batch_negatives=training_args.use_in_batch_negatives,
        use_in_batch_positives=training_args.use_in_batch_positives,
    )

    print(f"GPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    trainer = ContrastiveTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        contrastive_loss_fn=contrastive_loss_fn,
        contrastive_weight=training_args.contrastive_weight,
        classification_weight=training_args.classification_weight,
        **data_module,
    )

    trainer.train()
    print(f"GPU memory after training: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    print(f"Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    train()
