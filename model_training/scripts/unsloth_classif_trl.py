# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 true \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-SFT
"""



from unsloth import tokenizer_utils
def do_nothing(*args, **kwargs):
    pass
tokenizer_utils.fix_untrained_tokens = do_nothing

import torch
major_version, minor_version = torch.cuda.get_device_capability()
print(f"Major: {major_version}, Minor: {minor_version}")
from datasets import load_dataset
from trl import SFTTrainer
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, Trainer
from typing import Tuple
import warnings
from typing import Any, Dict, List, Union
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk import word_tokenize
from json import JSONEncoder

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from alignment import ScriptArguments, SFTConfig, get_dataset, get_model, get_tokenizer
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format
os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
DATA_PATH = os.environ['WORK']+ '/attributionBench_augmented_gen_mixedalltrain' #attributionBench_augmented_gen_from_hardpositives_only'  ###/attributionBench_lfqa_expertqa_dpo'




import torch.nn.functional as F
from tqdm import tqdm
import random
import json


logger = logging.getLogger(__name__)

prompt="""You will be given a claim and a document. Determine whether the claim is 'GROUNDED' or 'NOT GROUNDED' based on the document.A 'GROUNDED' claim is fully supported by the information provided in the document. It should be directly verifiable from the document. Only return the classification as the answer: 1 for 'GROUNDED' or 0 for 'NOT GROUNDED' without any explanation.

CLAIM: {} 

DOCUMENT: {}

CLASSIFICATION:{}"""

#experiment_name="unsloth_qwen_classif_4ep_attribench"


class NumpyEncoder(JSONEncoder):
    """
    Custom JSONEncoder that handles NumPy types.
    """
    def default(self, obj):
        # Convert NumPy arrays to lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Convert all NumPy numeric types (int, float, bool) to Python equivalents
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        
        # Let the base class handle other objects
        return JSONEncoder.default(self, obj)


def formatting_prompts_func(dataset_,tokenizer,max_seq_length=4096, without_label=False):
    texts = []
    for i,row  in dataset_.iterrows():
        claim = row['claim']
        doc = " ".join(row['references']) 
        # max_text_chars = (max_seq_length - 2000) * 4  # Conservative estimate
        
        # if len(doc+claim) > max_text_chars:
        #     doc = doc[:max_text_chars-len(claim)]
        
        # the csv is setup so that the label column corresponds exactly to the 3 classes defined above in the prompt (important)
        if without_label:
            label=""
        else:
            label=1 if row["attribution_label"]=="attributable" else 0
        text = prompt.format(claim, doc,label).rstrip()
        nb_dep=0
        prompt_len=tokenizer.encode(text, add_special_tokens=True) #word_tokenize(text)
        if  len(prompt_len)> (max_seq_length - 300):
            c_token=tokenizer.encode(claim, add_special_tokens=True) #word_tokenize(claim)
            d_token=tokenizer.encode(doc, add_special_tokens=True)  #word_tokenize(doc)
            #print("Deppasement:",  len(prompt_len), len(c_token),len(d_token))
            nb_dep+=1
            if len(c_token)> 300:
                new_size=max_seq_length - 300 - len(d_token) 
                truncated_claim= tokenizer.decode(c_token[:new_size]) #" ".join(c_token[:new_size])
                text = prompt.format(truncated_claim, doc,label).rstrip()
            if len(d_token)> max_seq_length - 300-len(c_token):
                new_size=max_seq_length - 300 - len(c_token) 
                truncated_doc= tokenizer.decode(d_token[:new_size]) #" ".join(d_token[:new_size])
                text = prompt.format(claim, truncated_doc,label).rstrip()
                #print("new text length",len(tokenizer.encode(text, add_special_tokens=True))) #len(word_tokenize(text)))
        texts.append(text)

    print("Number of rows with a context longer than ",max_seq_length ," is:",nb_dep)
    return texts


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    os.environ["WANDB_PROJECT"] = "unsloth_trl_qwen3_4B" #script_args.experiment_name

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
    
    ################
    # Load tokenizer
    ################
    #tokenizer = get_tokenizer(model_args, training_args)
    ############
    # Load model
    ############
    logger.info("*** Loading model ***")
    NUM_CLASSES = 2 # number of classes in the csv
    max_seq_length =4096  # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    model_name = os.environ['WORK'] + '/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/' #"Qwen/Qwen3-4B";
    load_in_4bit = False
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_args.model_name_or_path,load_in_4bit=load_in_4bit,
        max_seq_length = max_seq_length,
        dtype = dtype,
    )   

    """We now trim the classification head so the model can only say numbers 0-NUM_CLASSES and no other words. (We don't use 0 here but keeping it makes everything simpler)"""
    class_labels = [0, 1]
    number_token_ids = []
    #for i in range(0, NUM_CLASSES+1):
    for i in class_labels:
        number_token_ids.append(tokenizer.encode(str(i), add_special_tokens=False)[0])
    print(number_token_ids)
    # keep only the number tokens from lm_head
    par = torch.nn.Parameter(model.lm_head.weight[number_token_ids, :])

    old_shape = model.lm_head.weight.shape
    old_size = old_shape[0]
    print(par.shape)
    print(old_shape)

    model.lm_head.weight = par

    reverse_map = {value: idx for idx, value in enumerate(number_token_ids)} # will be used later to convert an idx from the old tokenizer to the new lm_head
    reverse_map

    from peft import LoftQConfig

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = [
            "lm_head", # can easily be trained because it now has a small size
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = True,  # We support rank stabilized LoRA
        # init_lora_weights = 'loftq',
        # loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1), # And LoftQ
    )
    logger.info("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    ################
    # Load datasets
    ################
 
    if script_args.local_data_path is not None:
        logger.info(f"Using provided local data path {script_args.local_data_path}")
        local_data_path = script_args.local_data_path
    else:
        logger.info(f"No local data path provided using {DATA_PATH}")
        local_data_path = DATA_PATH
    dataset =  datasets.load_from_disk(local_data_path)
    logger.info(f"USING DATASET FOUND  IN {local_data_path}")
    logger.info(dataset)

    train = dataset["train"] #, split=args.split, features=features)
    train_df = train.to_pandas()#.to_list()
    logger.info(train_df.iloc[0])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    train_df["label"]=train_df.apply(lambda x : 1 if x["attribution_label"]=="attributable" else 0,axis=1)
    logger.info(len(train_df))
    #dataset = get_dataset(script_args)
    train_df['text'] = formatting_prompts_func(train_df,tokenizer,max_seq_length=max_seq_length)
    #train_df=train_df.head(100)
    train_dataset = datasets.Dataset.from_pandas(train_df,preserve_index=False)


    #### test sets
    features = datasets.Features({
        'question': datasets.Value('string'),
        'claim': datasets.Value('string'),
        'claim_raw_string': datasets.Value('string'),
        'response': datasets.Value('string'),
        'references': datasets.Sequence(datasets.Value("string")),
        'citation_links': datasets.Sequence(datasets.Value("string")),
        'webpage_references': datasets.Sequence(datasets.Value("string")),
        'attribution_label': datasets.Value('string'),
        'src_dataset': datasets.Value('string'),
        'id': datasets.Value('string'),
        })
    data_path=os.environ['WORK']+ "/AttributionBench"
    data = datasets.load_from_disk(data_path)



    #val=data["dev"]
    #val_df = val.to_pandas()#.to_list()
    val_df["label"]=val_df.apply(lambda x : 1 if x["attribution_label"]=="attributable" else 0,axis=1)
    logger.info(len(val_df))
    logger.info(val_df.iloc[0])
    val_df['text'] = formatting_prompts_func(val_df,tokenizer,max_seq_length=max_seq_length)
    dev_dataset = datasets.Dataset.from_pandas(val_df,preserve_index=False)




    test = data["test"] 
    test_ood = data["test_ood"] 
    test = test.to_pandas()#.to_list()
    test["label"]=test.apply(lambda x : 1 if x["attribution_label"]=="attributable" else 0,axis=1)

    test_ood=test_ood.to_pandas()#.to_list()
    test_ood["label"]=test_ood.apply(lambda x : 1 if x["attribution_label"]=="attributable" else 0,axis=1)

    test["text"]=formatting_prompts_func(test,tokenizer,max_seq_length=max_seq_length,without_label=True)
    test_df=test

    test_ood["text"]=formatting_prompts_func(test_ood,tokenizer,max_seq_length=max_seq_length,without_label=True)

    class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
        def __init__(
            self,
            *args,
            mlm: bool = False,
            ignore_index: int = -100,
            **kwargs,
        ):
            super().__init__(*args, mlm=mlm, **kwargs)
            self.ignore_index = ignore_index

        def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
            batch = super().torch_call(examples)

            for i in range(len(examples)):
                # Find the last non-padding token
                last_token_idx = (batch["labels"][i] != self.ignore_index).nonzero()[-1].item()
                # Set all labels to ignore_index except for the last token
                batch["labels"][i, :last_token_idx] = self.ignore_index
                # If the last token in the text is, for example, "2", then this was processed with the old tokenizer into number_token_ids[2]
                # But we don't actually want this because number_token_ids[2] could be something like 27, which is now undefined in the new lm_head. So we map it to the new lm_head index.
                # if this line gives you a keyerror then increase max_seq_length
                batch["labels"][i, last_token_idx] = reverse_map[ batch["labels"][i, last_token_idx].item() ]


            return batch

    collator = DataCollatorForLastTokenLM(tokenizer=tokenizer)

    ############################
    # Initialize the SFT Trainer
    ############################
    # trainer = SFTTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset[script_args.dataset_train_split],
    #     eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
    #     processing_class=tokenizer,
    #     peft_config=get_peft_config(model_args),
    #     )
    trainer=SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset=dev_dataset,
        max_seq_length = max_seq_length,
        dataset_num_proc = 1,
        packing = False, # not needed because group_by_length is True
        args = SFTConfig(
            per_device_train_batch_size = 32,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            learning_rate = 1e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs",
            num_train_epochs = 2,
            report_to = "wandb",
            #report_to = "none",
            group_by_length = True,
            # per_device_train_batch_size = 32,
            # gradient_accumulation_steps = 4,
            # #do_eval= True,
            # eval_strategy= "epoch",
            # #eval_steps= 100,
            # save_strategy= "steps",
            # save_steps=100,
            # save_total_limit= 1,
            # #warmup_steps = 10,
            # warmup_ratio= 0.03,
            # learning_rate = 1e-5,
            # fp16 = not torch.cuda.is_bf16_supported(),
            # bf16 = torch.cuda.is_bf16_supported(),
            # logging_steps = 1,
            # optim = "adamw_8bit",
            # weight_decay = 0.0,
            # lr_scheduler_type = "cosine",
            # seed = 3407,
            # output_dir =training_args.output_dir,
            # num_train_epochs =training_args.num_train_epochs,
            # report_to = training_args.report_to, #"wandb",
            # run_name=script_args.experiment_name,
            # #report_to = "tensorboard",
            # group_by_length = True,
            # #load_best_model_at_end= True,
        ),
        data_collator=collator,
        dataset_text_field="text",
    )

    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")



    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    trainer_stats = trainer.train(resume_from_checkpoint=checkpoint)

    #@title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    logger.info(f"Peak reserved memory = {used_memory} GB.")
    logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    metrics = trainer_stats.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "model_name": training_args.hub_model_id if training_args.push_to_hub else None,
        "dataset_name": script_args.dataset_name,
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##################################
    # Unsloth Save/ and Eval
    ##################################
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    print()

    """### remake the old lm_head but with unused tokens having -1000 bias and 0 weights (improves compatibility with libraries like vllm)"""

    # Save the current (trimmed) lm_head and bias
    trimmed_lm_head = model.lm_head.weight.data.clone()
    trimmed_lm_head_bias = model.lm_head.bias.data.clone() if hasattr(model.lm_head, "bias") and model.lm_head.bias is not None else torch.zeros(len(number_token_ids), device=trimmed_lm_head.device)

    # Create a new lm_head with shape [old_size, hidden_dim]
    hidden_dim = trimmed_lm_head.shape[1]
    new_lm_head = torch.full((old_size, hidden_dim), 0, dtype=trimmed_lm_head.dtype, device=trimmed_lm_head.device)
    new_lm_head_bias = torch.full((old_size,), -1000.0, dtype=trimmed_lm_head_bias.dtype, device=trimmed_lm_head_bias.device)

    # Fill in the weights and bias for the allowed tokens (number_token_ids)
    for new_idx, orig_token_id in enumerate(number_token_ids):
        new_lm_head[orig_token_id] = trimmed_lm_head[new_idx]
        new_lm_head_bias[orig_token_id] = trimmed_lm_head_bias[new_idx]

    # Update the model's lm_head weight and bias
    with torch.no_grad():
        new_lm_head_module = torch.nn.Linear(hidden_dim, old_size, bias=True, device=model.device)
        new_lm_head_module.weight.data.copy_(new_lm_head)
        new_lm_head_module.bias.data.copy_(new_lm_head_bias)
        model.lm_head.modules_to_save["default"] = new_lm_head_module

    logger.info(f"Remade lm_head: shape = {model.lm_head.weight.shape}. Allowed tokens: {number_token_ids}")



    """### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
    """

    # # Merge to 16bit

    logger.info(f"Model 16bit saved to {training_args.output_dir}")
    model.save_pretrained_merged(training_args.output_dir+"_16bit", tokenizer, save_method = "merged_16bit",push_to_hub=False)
    # if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

    # """### GGUF / llama.cpp Conversion
    # To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
    # """

    # # Save to 8bit Q8_0
    #model.save_pretrained_gguf("qwen_classif_trained_8bit", tokenizer,push_to_hub=False)
    # if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

    # # Save to 16bit GGUF
    #model.save_pretrained_gguf("qwen_classif_trained_16bit", tokenizer, quantization_method = "f16",push_to_hub=False)
    # if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

    # # Save to q4_k_m GGUF
    #model.save_pretrained_gguf("qwen_classif_trained_q4_k_m", tokenizer, quantization_method = "q4_k_m",push_to_hub=False)
    # if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")


    ##########
    # Evaluate
    ##########
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    #############
    # Inference
    #############
    logger.info("*** Running inference ***")
    """# Batched Inference on Validation Set"""

    ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
    sys.path.append(ROOT_PATH)

    from RAGnRoll.tools.eval_tools import read_json_files_from_folder, compute_metrics

    if script_args.do_inference:
        display = 2
        batch_size = 5
        def run_inference(df, batch_size = 5, display = 2):
            # Prepare inference prompt
            inference_prompt_template = prompt.split("CLASSIFICATION:{}")[0] + "CLASSIFICATION: "
            # Sort validation set by length for efficient batching
            df['token_length'] = df['text'].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
            df_sorted = df.sort_values(by='token_length').reset_index(drop=True)
            # Clean up
            if 'token_length' in df:
                del df['token_length']
            if 'token_length' in df_sorted:
                del df_sorted['token_length']
            device = model.device
            correct = 0
            results = []
            with torch.inference_mode():
                for i in tqdm(range(0, len(df_sorted), batch_size), desc="Evaluating"):
                    batch = df_sorted.iloc[i:i+batch_size]
                    prompts = list(batch["text"]) # [e  for e in batch["text"]]# inference_prompt_template.format(e["claim"]," ".join(e["references"])) for e in batch["text"]] #.split("CLASSIFICATION:")[0] + "CLASSIFICATION: "
                    #print("EXAMPLE PROMPT:",prompts[0])
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to(device)
                    logits = model(**inputs).logits
                    last_idxs = inputs.attention_mask.sum(1) - 1
                    last_logits = logits[torch.arange(len(batch)), last_idxs, :]
                    probs_all = F.softmax(last_logits, dim=-1)
                    probs = probs_all[:, number_token_ids] # only keep the logits for the number tokens
                    preds = torch.argmax(probs, dim=-1).cpu().numpy() # looks like [1 1 1 1 3 1 3 1 3 1 1 1 1 2 2 3]

                    true_labels = batch['label'].tolist()
                    correct += sum([p == t for p, t in zip(preds, true_labels)])
                    # Store a few samples for display
                    for j in range(len(batch)):
                        item = batch.iloc[j].to_dict() 
                        #item['text']=item[:]
                        if (true_labels[j] == 1 and item["attribution_label"] != "attributable") or (true_labels[j] == 0 and item["attribution_label"] != "not attributable"):
                            print("true labels mismatch:", true_labels[j], item["attribution_label"])
                        item['auto_score']=  preds[j].item()
                        item['accuracy']= 1 if (preds[j] == true_labels[j]).item() else 0
                        item['all_probs']= probs[j].float().cpu().numpy().tolist()
                        item['logit']=item['all_probs'][preds[j]]
                        item['references']=item['references'].tolist()
                        item['citation_links']=item['citation_links'].tolist()
                        item['webpage_references']=item['webpage_references'].tolist()
                        
                        #print("Logit for item",j,item['logit'], type(item['logit']),item['all_probs'],type(item['all_probs']))
                        #print(item.keys())
                        results.append(item)
                        #     {
                        #     "text": batch['text'].iloc[j][:200],
                        #     "true": true_labels[j],
                        #     "pred": preds[j],
                        #     "probs": probs[j][1:].float().cpu().numpy(), # ignore prob for class 0 and convert from tensor to float
                        #     "ok": preds[j] == true_labels[j]
                        # })

            accuracy = 100 * correct / len(df_sorted)
            print(f"\nValidation accuracy: {accuracy:.2f}% ({correct}/{len(df_sorted)})")
            return results

        logger.info("EVALUATING TEST SET")

        inference_dir = os.path.join(training_args.output_dir, "inference")
        try:
            os.makedirs(inference_dir, exist_ok=True)
            print(f"Result folder created successfully at: {inference_dir}")
        except OSError as e:
            print(f"Error creating directory: {e}")

        results= run_inference(test_df)

        print("\n--- Random samples ---")
        for s in random.sample(results, min(display, len(results))):
            print(f"\nText: {s['text']}")
            print(f"True: {s['attribution_label']}  Pred: {s['auto_score']} {'✅' if s['accuracy'] else '❌'}")
            print("Probs:", ", ".join([f"{k}: {v:.3f}" for k, v in enumerate(s['all_probs'], start=0)]))


        print(results[0])
        all_scores=compute_metrics(results,prediction_column="auto_score", scoredlabels=True)
        saving_dict={"data":results, "overall":all_scores[0].to_dict('index'), "per_data_src":all_scores[1].to_dict('index')}

        results_file = os.path.join(inference_dir, "test_eval.json")
        logger.info(f"Saving Result to {results_file}")
        with open(results_file, "w") as f:
            json.dump(saving_dict, f, indent=4,cls=NumpyEncoder)

        #### ood
        logger.info("EVALUATING TEST OUT OF DISTRIBUTION")

        results= run_inference(test_ood)

        print("\n--- Random samples ---")
        for s in random.sample(results, min(display, len(results))):
            print(f"\nText: {s['text']}")
            print(f"True: {s['attribution_label']}  Pred: {s['auto_score']} {'✅' if s['accuracy'] else '❌'}")
            print("Probs:", ", ".join([f"{k}: {v:.3f}" for k, v in enumerate(s['all_probs'], start=0)]))


        print(results[0])
        all_scores=compute_metrics(results,prediction_column="auto_score", scoredlabels=True)
        saving_dict={"data":results, "overall":all_scores[0].to_dict('index'), "per_data_src":all_scores[1].to_dict('index')}

        results_file = os.path.join(inference_dir, "test_eval_ood.json")
        logger.info(f"Saving Result to {results_file}")
        with open(results_file, "w") as f:
            json.dump(saving_dict, f, indent=4,cls=NumpyEncoder)


    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
