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

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from alignment import ScriptArguments, SFTConfig, get_dataset, get_model, get_tokenizer
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
from transformers import DataCollatorForSeq2Seq

os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
DATA_PATH = os.environ['WORK']+ '/attributionBench_augmented_gen_mixedalltrain' #attributionBench_augmented_gen_from_hardpositives_only'  ###/attributionBench_lfqa_expertqa_dpo'


logger = logging.getLogger(__name__)

def preprocess_function(examples, tokenizer):
    """
    Preprocess function for T5 text classification.
    T5 uses text-to-text format, so we convert labels to text.
    """
    # Add task prefix for T5 (important for T5's architecture)
    task_prefix = "classify: "
    
    # Extract input text from messages
    inputs = [
        task_prefix + (t[0]["content"] if t[0]["role"] == "user" else t[1]["content"])
        for t in examples["messages"]
    ]
    
    # Extract target labels as strings
    targets = [
        str(t[1]["content"]) if t[1]["role"] == "assistant" else str(t[0]["content"])
        for t in examples["messages"]
    ]
    
    # Tokenize inputs (encoder)
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding=False  # Let data collator handle padding
    )

    # Tokenize targets (decoder)
    labels = tokenizer(
        text_target=targets,  # Use text_target parameter for T5
        max_length=10,
        truncation=True,
        padding=False  # Let data collator handle padding
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """
    Compute accuracy metrics for text classification.
    For T5, predictions are generated token IDs that need to be decoded.
    """
    predictions, label_ids = eval_preds
    
    # Decode predictions and labels
    # Replace -100 in labels (used for padding) with pad_token_id
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Calculate accuracy
    correct = sum([pred == label for pred, label in zip(decoded_preds, decoded_labels)])
    accuracy = correct / len(decoded_labels) if len(decoded_labels) > 0 else 0
    
    # Optional: Log some examples for debugging
    if len(decoded_preds) > 0:
        logger.info(f"Sample predictions: {decoded_preds[:5]}")
        logger.info(f"Sample labels: {decoded_labels[:5]}")
    
    return {
        "accuracy": accuracy,
        "exact_match": accuracy  # Same as accuracy for classification
    }


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

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
    print(dataset)

    #dataset = get_dataset(script_args)
    
    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)


    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        # Remove original columns to only keep tokenized inputs and labels for training
        remove_columns=dataset[script_args.dataset_train_split].column_names, 
        )

    ############
    # Load model
    ############
    logger.info("*** Loading T5 model for conditional classification ***")
    model = get_model(model_args, training_args)

    # Set generation config for T5
    model.config.max_length = 10  # Max output length for classification
    #model.config.num_beams = 1  # Use greedy decoding for speed
    
    # Data collator for seq2seq models
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100  # Standard for ignoring padding in loss
    )
    #data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # Create compute_metrics function with tokenizer
    def compute_metrics_with_tokenizer(eval_preds):
        return compute_metrics(eval_preds, tokenizer)



    # if tokenizer.chat_template is None:
    #     logger.info("No chat template provided, using ChatML.")
    #     model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

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
    # )


    # trainer = SFTTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset[script_args.dataset_train_split], # Use tokenized dataset
    #     eval_dataset=(tokenized_dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None), # Use tokenized dataset
    #     # Change: Do not pass processing_class=tokenizer here if using DataCollator
    #     # If SFTTrainer insists on a tokenizer, it's for internal formatting/packing, 
    #     # but the primary input shaping is done by the data_collator and .map().
    #     data_collator=data_collator, # Add data collator
    #     compute_metrics=compute_metrics, # Add metrics
    #     peft_config=get_peft_config(model_args), # Keep if using PEFT
    # )
    # model.config.max_length = 10 
        
    #     # Set the max length for the ENCODER input
    #     # Assuming the value in your YAML 'max_seq_length' is the desired input length
    # max_input_length = 2048

     ############################
    # Initialize Seq2Seq Trainer
    ############################
    # Convert SFTConfig to Seq2SeqTrainingArguments if needed
    seq2seq_training_args = Seq2SeqTrainingArguments(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        warmup_steps=training_args.warmup_steps,
        logging_steps=training_args.logging_steps,
        eval_strategy=training_args.eval_strategy,
        eval_steps=training_args.eval_steps if training_args.eval_strategy == "steps" else None,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps if training_args.save_strategy == "steps" else None,
        save_total_limit=training_args.save_total_limit,
        load_best_model_at_end=training_args.load_best_model_at_end,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        gradient_checkpointing=training_args.gradient_checkpointing,
        seed=training_args.seed,
        report_to=training_args.report_to,
        push_to_hub=training_args.push_to_hub,
        hub_model_id=training_args.hub_model_id if training_args.push_to_hub else None,
        # Critical for T5: enable generation during evaluation
        predict_with_generate=True,
        generation_max_length=10,  # Max length for generated labels
        #generation_num_beams=1,  # Greedy decoding for speed
    )

    trainer = Seq2SeqTrainer(
        model=model,
        # CHANGE 2: T5 is a generative model, so ensure you enable generation for evaluation
        args=seq2seq_training_args, # Assuming training_args has predict_with_generate=True (or is Seq2SeqTrainingArguments)
        train_dataset=tokenized_dataset[script_args.dataset_train_split],
        eval_dataset=(tokenized_dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        tokenizer=tokenizer, # Pass tokenizer for Seq2SeqTrainer/DataCollator
        data_collator=data_collator, 
        compute_metrics=compute_metrics_with_tokenizer,
        # peft_config is typically passed via model loading or a custom Trainer/SFTTrainer
        # If you are using PEFT, ensure it works correctly with Seq2SeqTrainer.
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
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

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

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
