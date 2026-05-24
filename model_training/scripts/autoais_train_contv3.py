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
from peft import LoraConfig, get_peft_model, TaskType
import random
from collections import defaultdict

os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
os.environ['WANDB_MODE'] = 'offline'

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
SHOW_BATCH_SIZE = 0

T5_TYPES = ["claim_erronous_change", "claim_numerical_mismatch", "modify_passage-add_relevant_to_claim",
            "claim_combine_facts", "claim_add_to_the_claim_contradicting_info", "modify_passage-add_contradiction",
            "modify_passage-add_conflicting_sources", "claim_infer_claim", "claim_over_infer_claim"]


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/t5_xxl_true_nli_mixture")


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
        metadata={"help": "Use decoder hidden state (True) or encoder first token (False) for contrastive learning"}
    )
    use_in_batch_negatives: bool = field(
        default=True,
        metadata={"help": "Fall back to in-batch negatives when explicit neg are missing"}
    )
    use_in_batch_positives: bool = field(
        default=False,
        metadata={"help": "Uses in batch positives when positives are missing"}
    )
    remove_unused_columns: bool = field(default=False) 

class HardNegativeContrastiveLoss(nn.Module):
    """
    Contrastive loss with explicit hard positives/negatives, with per-anchor dispatch:

      pos + neg  -> SupCon over explicit pairs
      neg only   -> repulsion (minimize cosine sim to negatives);
                    if use_in_batch_negatives, also pull toward in-batch same-label anchors
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
            print("Degenerated batch, empty per_anchor_losses")
            self.degenerate_batch_count += 1
            logging.warning(
                f"[ContrastiveLoss] Degenerate batch "
                f"({self.degenerate_batch_count}/{self.total_batch_count}): "
                f"no valid anchors — falling back to classification loss."
            )
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(per_anchor_losses).mean()

    # ------------------------------------------------------------------

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
            return None  # no negatives available, skip

        # Case 3: explicit neg only
        if has_neg:
            print(" Neg only ")
            if self.use_in_batch_positives:
                # Upgrade to full SupCon if in-batch positives are available
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

        return None  # fall back to classification loss only

    def _supcon(
        self,
        a: torch.Tensor,    # [1, D]
        pos: torch.Tensor,  # [p, D]  p >= 1
        neg: torch.Tensor,  # [n, D]  n >= 1
    ) -> torch.Tensor:
        t = self.temperature
        sim_pos = (a @ pos.T).squeeze(0) / t   # [p]
        sim_neg = (a @ neg.T).squeeze(0) / t   # [n]
        all_sim = torch.cat([sim_pos, sim_neg])  # [p+n]

        shift = all_sim.max().detach()
        log_denom = torch.log(torch.exp(all_sim - shift).sum() + 1e-8)
        log_probs = (sim_pos - shift) - log_denom  # [p]
        return -log_probs.mean()

    def _inbatch_same_label(
        self, i: int, anchor_embeds: torch.Tensor, anchor_labels: torch.Tensor
    ) -> torch.Tensor:
        mask = anchor_labels == anchor_labels[i]
        mask[i] = False
        return anchor_embeds[mask]  # [k, D]

    def _inbatch_diff_label(
        self, i: int, anchor_embeds: torch.Tensor, anchor_labels: torch.Tensor
    ) -> torch.Tensor:
        mask = anchor_labels != anchor_labels[i]
        return anchor_embeds[mask]  # [k, D]


# def get_decoder_embedding(model, input_ids, attention_mask):
#     """
#     Get sentence embedding from T5 decoder's first-step hidden state.
#     """
#     model_to_use = model.module if hasattr(model, "module") else model

#     encoder_outputs = model_to_use.encoder(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         return_dict=True
#     )

#     decoder_start_token_id = model_to_use.config.decoder_start_token_id
#     decoder_input_ids = torch.full(
#         (input_ids.shape[0], 1),
#         decoder_start_token_id,
#         dtype=torch.long,
#         device=input_ids.device
#     )

#     decoder_outputs = model_to_use.decoder(
#         input_ids=decoder_input_ids,
#         encoder_hidden_states=encoder_outputs.last_hidden_state,
#         encoder_attention_mask=attention_mask,
#         return_dict=True
#     )

#     return decoder_outputs.last_hidden_state[:, 0, :]  # [B, D]


class ContrastiveDataset(Dataset):
    """
    Dataset that loads anchor + optional explicit positives/negatives for contrastive learning.
    Expected dataset format: {anchor: {...}, positives: [{...}, ...], negatives: [{...}, ...]}
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

        self.data = self.load_dataset(split, data_args)

        logging.info(f"Loaded {len(self.data)} examples for {split}")
        if self.use_contrastive:
            logging.info(f"Contrastive learning enabled: sampling up to "
                         f"{data_args.num_positives} pos, {data_args.num_negatives} neg")

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
        features = Features(
            {
                "question": Value("string"),
                "claim": Value("string"),
                "claim_raw_string": Value("string"),
                "response": Value("string"),
                "references": datasets.Sequence(Value("string")),
                "citation_links": datasets.Sequence(Value("string")),
                "webpage_references": datasets.Sequence(Value("string")),
                "attribution_label": Value("string"),
                "src_dataset": Value("string"),
                "id": Value("string"),
            }
        )
        data_path = os.environ['WORK'] + "/AttributionBench"
        data = datasets.load_from_disk(data_path)
        if split in ["stanford_dev", "attributedqa_dev", "hagrid_dev", "expertqa_dev"]:
            dataset = data["dev"]
        elif split == "train":
            data_path = os.environ['WORK'] + "/" + data_args.dataset_version
            data = datasets.load_from_disk(data_path)
            dataset = data[split]
            #dataset = self.stratified_shuffle(dataset, 4)
            # for ishuf in range(4):
            #     anchor = dataset[ishuf]['anchor'] if "anchor" in dataset[ishuf] else dataset[ishuf]
            #     print(anchor["attribution_label"])
        else:
            data_path = os.environ['WORK'] + "/AttributionBench"
            data = datasets.load_from_disk(data_path)
            dataset = data[split]
        return dataset

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
                print("Empty references", example)

            if have_question and have_response:
                input_template = "### Input:\nQuestion: {}\n\nClaim: {}\n\nResponse: {}\n\nReference: {}\n\n### Output:"
                input = input_template.format(query, answer, response, documents_concatenation)
            elif have_question and not have_response:
                input_template = "premise: {} hypothesis: {}"
                input = input_template.format(documents_concatenation, " ".join(query, answer))
            elif not have_question and have_response:
                input_template = "### Input:\nClaim: {}\n\nResponse: {}\n\nReference: {}\n\n### Output:"
                input = input_template.format(answer, response, documents_concatenation)
            else:
                input_template = "premise: {} hypothesis: {}"
                input = input_template.format(documents_concatenation, answer)

            instructions = json.load(open(self.data_args.template_path))
            return input

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

        return format_prompt(
            example,
            have_question=have_question,
            have_response=have_response,
            prompt_name=self.data_args.template,
        )

    def tokenize_example(self, text, is_target=False):
        if is_target:
            token_ids = self.tokenizer(
                text_target=text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids[0]
            token_ids = torch.where(token_ids == self.tokenizer.pad_token_id, -100, token_ids)
            return token_ids
        else:
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            return encoding.input_ids[0], encoding.attention_mask[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        anchor = example['anchor'] if "anchor" in example else example

        anchor_text = self.process_function(anchor)
        anchor_input_ids, anchor_attention_mask = self.tokenize_example(anchor_text)

        anchor_label = "1" if str(anchor.get('attribution_label', '')) == "attributable" else "0"
        anchor_labels = self.tokenize_example(anchor_label, is_target=True)

        result = {
            "input_ids": anchor_input_ids,          # [L_a]
            "attention_mask": anchor_attention_mask, # [L_a]
            "labels": anchor_labels,                 # [L_t]
        }

        if self.use_contrastive:
            pos_input_ids_list, pos_attention_mask_list, pos_labels_list = [], [], []
            neg_input_ids_list, neg_attention_mask_list, neg_labels_list = [], [], []
            print("processing Positives and Negatives in Data module")
            for pos_ex in (example.get("positives") or [])[:self.data_args.num_positives]:
                text = self.process_function(pos_ex)
                ids, mask = self.tokenize_example(text)
                pos_input_ids_list.append(ids)
                pos_attention_mask_list.append(mask)
                lbl = "1" if str(pos_ex.get("attribution_label", "")) == "attributable" else "0"
                pos_labels_list.append(self.tokenize_example(lbl, is_target=True))

            for neg_ex in (example.get("negatives") or [])[:self.data_args.num_negatives]:
                text = self.process_function(neg_ex)
                ids, mask = self.tokenize_example(text)
                neg_input_ids_list.append(ids)
                neg_attention_mask_list.append(mask)
                lbl = "1" if str(neg_ex.get("attribution_label", "")) == "attributable" else "0"
                neg_labels_list.append(self.tokenize_example(lbl, is_target=True))
            print("Len Pos and negative in Data module:", len(pos_attention_mask_list),len(neg_input_ids_list))

            result["pos_input_ids"] = pos_input_ids_list
            result["pos_attention_masks"] = pos_attention_mask_list
            result["pos_labels"] = pos_labels_list
            result["neg_input_ids"] = neg_input_ids_list
            result["neg_attention_masks"] = neg_attention_mask_list
            result["neg_labels"] = neg_labels_list

        return result


@dataclass
class ContrastiveDataCollator:
    """Collator for contrastive learning batches. Flattens variable-length pos/neg lists."""
    tokenizer: transformers.PreTrainedTokenizer
    use_contrastive: bool = True

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad anchors
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
        )  # [B, L_t]

        batch = {
            "input_ids": anchor_input_ids,
            "attention_mask": anchor_attention_masks,
            "labels": anchor_labels,
        }

        if not self.use_contrastive:
            print("!!!! X !!!!! Not using contrastive")
            return batch

        # Flatten all positives/negatives across the batch, tracking which anchor owns each
        all_pos_ids, all_pos_masks, all_pos_labels, pos_anchor_idx = [], [], [], []
        all_neg_ids, all_neg_masks, all_neg_labels, neg_anchor_idx = [], [], [], []

        for batch_i, inst in enumerate(instances):
            for ids, mask, lbl in zip(
                inst.get("pos_input_ids", []),
                inst.get("pos_attention_masks", []),
                inst.get("pos_labels", []),
            ):
                all_pos_ids.append(ids)
                all_pos_masks.append(mask)
                all_pos_labels.append(lbl)
                pos_anchor_idx.append(batch_i)
            for ids, mask, lbl in zip(
                inst.get("neg_input_ids", []),
                inst.get("neg_attention_masks", []),
                inst.get("neg_labels", []),
            ):
                all_neg_ids.append(ids)
                all_neg_masks.append(mask)
                all_neg_labels.append(lbl)
                neg_anchor_idx.append(batch_i)
        print("Len Pos and negative in Collator:", len(all_pos_ids), len(all_neg_ids))

        if all_pos_ids:
            batch["pos_input_ids"] = torch.nn.utils.rnn.pad_sequence(
                all_pos_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            batch["pos_attention_masks"] = torch.nn.utils.rnn.pad_sequence(
                all_pos_masks, batch_first=True, padding_value=0
            )
            batch["pos_labels"] = torch.nn.utils.rnn.pad_sequence(
                all_pos_labels, batch_first=True, padding_value=IGNORE_INDEX
            )
            batch["pos_anchor_idx"] = torch.tensor(pos_anchor_idx, dtype=torch.long)
        else:
            batch["pos_input_ids"] = torch.zeros(0, 1, dtype=torch.long)
            batch["pos_attention_masks"] = torch.zeros(0, 1, dtype=torch.long)
            batch["pos_labels"] = torch.zeros(0, 1, dtype=torch.long)
            batch["pos_anchor_idx"] = torch.zeros(0, dtype=torch.long)

        if all_neg_ids:
            batch["neg_input_ids"] = torch.nn.utils.rnn.pad_sequence(
                all_neg_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            batch["neg_attention_masks"] = torch.nn.utils.rnn.pad_sequence(
                all_neg_masks, batch_first=True, padding_value=0
            )
            batch["neg_labels"] = torch.nn.utils.rnn.pad_sequence(
                all_neg_labels, batch_first=True, padding_value=IGNORE_INDEX
            )
            batch["neg_anchor_idx"] = torch.tensor(neg_anchor_idx, dtype=torch.long)
        else:
            batch["neg_input_ids"] = torch.zeros(0, 1, dtype=torch.long)
            batch["neg_attention_masks"] = torch.zeros(0, 1, dtype=torch.long)
            batch["neg_labels"] = torch.zeros(0, 1, dtype=torch.long)
            batch["neg_anchor_idx"] = torch.zeros(0, dtype=torch.long)
        print("Len Pos and negative in Collator:", len(batch["neg_input_ids"]), len(batch["pos_input_ids"]))

        return batch


class ContrastiveTrainer(Seq2SeqTrainer):
    """
    Custom trainer that combines classification loss with hard-negative contrastive loss.
    """
    def __init__(self, *args, contrastive_loss_fn=None, contrastive_weight=0.5,
                 classification_weight=1.0, use_decoder_embedding=True,use_contrastive: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss_fn = contrastive_loss_fn
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        self.use_decoder_embedding = use_decoder_embedding
        self.use_contrastive=use_contrastive
        self.custom_loss_tracker = {'classification': [], 'contrastive': []}

    # def _get_embeddings(
    #     self,
    #     model,
    #     input_ids: torch.Tensor,      # [N, L]
    #     attention_mask: torch.Tensor,  # [N, L]
    # ) -> torch.Tensor:
    #     """Forward pass to extract embeddings for an arbitrary batch of sequences. Returns [N, D]."""
    #     if input_ids.shape[0] == 0:
    #         D = (model.module if hasattr(model, "module") else model).config.d_model
    #         return torch.zeros(0, D, device=input_ids.device)

        
    #     if self.use_decoder_embedding:
    #         return get_decoder_embedding(model, input_ids, attention_mask)
    #     else:
    #         model_inner = model.module if hasattr(model, "module") else model
    #         enc_out = model_inner.encoder(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             return_dict=True,
    #         )
    #         return enc_out.last_hidden_state[:, 0, :]


    def _get_embeddings(
        self,
        model,
        input_ids: torch.Tensor,       # [N, L]
        attention_mask: torch.Tensor,  # [N, L]
        labels: torch.Tensor,          # [N, L_t]  same format as anchor labels
    ) -> torch.Tensor:
        """Forward pass to extract embeddings for an arbitrary batch of sequences. Returns [N, D]."""
        if input_ids.shape[0] == 0:
            D = (model.module if hasattr(model, "module") else model).config.d_model
            return torch.zeros(0, D, device=input_ids.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        if self.use_decoder_embedding:
            return outputs.decoder_hidden_states[-1][:, 0, :]
        else:
            return outputs.encoder_last_hidden_state[:, 0, :]

         


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop contrastive tensors — T5ForConditionalGeneration doesn't accept them
        pos_input_ids = inputs.pop("pos_input_ids", None)
        pos_attention_masks = inputs.pop("pos_attention_masks", None)
        pos_labels = inputs.pop("pos_labels", None)
        pos_anchor_idx = inputs.pop("pos_anchor_idx", None)
        neg_input_ids = inputs.pop("neg_input_ids", None)
        neg_attention_masks = inputs.pop("neg_attention_masks", None)
        neg_labels = inputs.pop("neg_labels", None)
        neg_anchor_idx = inputs.pop("neg_anchor_idx", None)

        # Main forward pass on anchors
        outputs = model(**inputs, output_hidden_states=True)
        classification_loss = outputs.loss
        if not self.use_contrastive:
            return classification_loss
        # Decode labels to get 0/1 anchor labels for in-batch fallback
        labels_decoded = inputs["labels"].clone()
        # _tok = getattr(self, 'processing_class', None) or self.tokenizer
        # labels_decoded[labels_decoded == IGNORE_INDEX] = _tok.pad_token_id
        # labels_text = _tok.batch_decode(labels_decoded, skip_special_tokens=True)
        labels_decoded[labels_decoded == IGNORE_INDEX] = self.tokenizer.pad_token_id
        labels_text = self.tokenizer.batch_decode(labels_decoded, skip_special_tokens=True)
        anchor_labels = torch.tensor(
            [int(t.strip() == "1") for t in labels_text],
            device=inputs["labels"].device
        )  # [B]

        # Anchor embeddings from main pass
        if self.use_decoder_embedding:
            anchor_embeds = outputs.decoder_hidden_states[-1][:, 0, :]  # [B, D]
        else:
            anchor_embeds = outputs.encoder_last_hidden_state[:, 0, :]  # [B, D]

        # Compute contrastive loss if the loss function is set
        contrastive_loss = torch.tensor(0.0, device=anchor_embeds.device)

        if self.contrastive_loss_fn is not None and self.use_contrastive:
            print("Computing Contrastive Loss")
            device = anchor_embeds.device
            D = anchor_embeds.shape[-1]

            # Always call _get_embeddings on every rank (even with a dummy anchor
            # input when pos/neg is absent) so all ranks participate in the same
            # FSDP all-gather forward passes and never deadlock.
            _has_pos = pos_input_ids is not None and pos_input_ids.shape[0] > 0
            _pos_ids   = pos_input_ids.to(device)        if _has_pos else inputs["input_ids"][0:1]
            _pos_masks = pos_attention_masks.to(device)  if _has_pos else inputs["attention_mask"][0:1]
            _pos_lbls  = pos_labels.to(device)           if _has_pos else inputs["labels"][0:1]
            _pos_out   = self._get_embeddings(model, _pos_ids, _pos_masks, _pos_lbls)
            pos_embeds     = _pos_out if _has_pos else torch.zeros(0, D, device=device)
            pos_anchor_idx = pos_anchor_idx.to(device) if _has_pos else torch.zeros(0, dtype=torch.long, device=device)

            _has_neg = neg_input_ids is not None and neg_input_ids.shape[0] > 0
            _neg_ids   = neg_input_ids.to(device)        if _has_neg else inputs["input_ids"][0:1]
            _neg_masks = neg_attention_masks.to(device)  if _has_neg else inputs["attention_mask"][0:1]
            _neg_lbls  = neg_labels.to(device)           if _has_neg else inputs["labels"][0:1]
            _neg_out   = self._get_embeddings(model, _neg_ids, _neg_masks, _neg_lbls)
            neg_embeds     = _neg_out if _has_neg else torch.zeros(0, D, device=device)
            neg_anchor_idx = neg_anchor_idx.to(device) if _has_neg else torch.zeros(0, dtype=torch.long, device=device)

            contrastive_loss = self.contrastive_loss_fn(
                anchor_embeds,   # [B, D]
                pos_embeds,      # [P_total, D]
                neg_embeds,      # [N_total, D]
                pos_anchor_idx,  # [P_total]
                neg_anchor_idx,  # [N_total]
                anchor_labels,   # [B]
            )
        else:
            print("Not using  Contrastive Loss: ",self.use_contrastive)
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
        if len(self.custom_loss_tracker['classification']) > 0:
            logs["class_loss"] = (sum(self.custom_loss_tracker['classification']) /
                                  len(self.custom_loss_tracker['classification']))
            logs["cont_loss"] = (sum(self.custom_loss_tracker['contrastive']) /
                                 len(self.custom_loss_tracker['contrastive']))
            self.custom_loss_tracker['classification'] = []
            self.custom_loss_tracker['contrastive'] = []
        super().log(logs, *args, **kwargs)


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
    print("train preparation")
    train_dataset = ContrastiveDataset(
        tokenizer=tokenizer, data_args=data_args, split="train"
    )

    data_collator = ContrastiveDataCollator(
        tokenizer=tokenizer,
        use_contrastive=data_args.use_contrastive
    )

    eval_data_args = copy.deepcopy(data_args)
    eval_data_args.use_contrastive = False

    print("dev preparation")
    eval_dataset = ContrastiveDataset(
        tokenizer=tokenizer, data_args=eval_data_args, split="dev"
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
    training_args.report_to = "wandb"
    training_args.remove_unused_columns = False
    with open(data_args.template_path) as f:
        template = json.load(f)

    model = transformers.T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # model = transformers.T5ForConditionalGeneration.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    # )
    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_modules=["q", "v"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type=TaskType.SEQ_2_SEQ_LM
    # )

    # model = get_peft_model(model, lora_config)
    #model.print_trainable_parameters()

    global tokenizer
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    contrastive_loss_fn = HardNegativeContrastiveLoss(
        temperature=training_args.contrastive_temperature,
        use_in_batch_negatives=training_args.use_in_batch_negatives,
        use_in_batch_positives=training_args.use_in_batch_positives,
    )

    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    trainer = ContrastiveTrainer(
        model=model,
        #processing_class=tokenizer,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        args=training_args,
        contrastive_loss_fn=contrastive_loss_fn,
        contrastive_weight=training_args.contrastive_weight,
        classification_weight=training_args.classification_weight,
        use_decoder_embedding=training_args.use_decoder_embedding,
        use_contrastive=data_args.use_contrastive,
        **data_module,
    )

    trainer.train()
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    print(f"Model Saved to : {training_args.output_dir}")


if __name__ == "__main__":
    train()
