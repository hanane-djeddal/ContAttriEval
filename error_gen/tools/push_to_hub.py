import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
from scipy.stats import pearsonr
from nltk import word_tokenize
import random

import datasets
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'

def format_dpo(data):
    updated_data = []
    for row in data:
        content_user="premise: {} \n\n hypothesis: {}".format(" ".join(row["references"]), row["claim"])
        chosen_content="1" if row["attribution_label"] == "attributable" else "0"
        rejected_content="0" if row["attribution_label"] == "attributable" else "1"
        row["chosen"]=[{"content":content_user,"role": "user"},{"content":chosen_content,"role": "assistant"}]
        row["rejected"]=[{"content":content_user,"role": "user"},{"content":rejected_content,"role": "assistant"}]
        row["messages"]=row["chosen"]
        row["score_chosen"]=8
        row["score_rejected"]=2.5
        updated_data.append(row)
    return updated_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    
    parser.add_argument(
        "--dev_file",type=str, default=None, help="file dev split "
    )    
    
    parser.add_argument(
        "--og_dev",action="store_true", help="og dev split "
    )   
    
    parser.add_argument(
        "--subset",type=str, default="subset_balanced", help=" dev split "
    )
    
    parser.add_argument(
        "--train_file",type=str, default=None, help="file dev split "
    )
    
    parser.add_argument(
        "--hf_dataset_name",type=str, default=None, help="file dev split "
    )
    
    parser.add_argument(
        "--save_local",action="store_true", help="prediction is a score"
    )
    parser.add_argument(
        "--shuffle",action="store_true", help="file dev split "
    ) 

    parser.add_argument(
        "--filter_error_type",type=str, default=None, help="filter "
    )  
    parser.add_argument(
        "--contrastive",action="store_true", help="contrastive "
    )  
    
   
    args = parser.parse_args()

    #################
    # Logging params
    #################
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)} - {parser.get_default(arg)}")
    if args.train_file:
        with open(args.train_file, 'r') as file:
            train = json.load(file)
            if isinstance(train, dict) and "data" in train.keys():
                train=train["data"]
        if args.shuffle:
              print("Before Shuffle", train[0])
              random.shuffle(train)
              print("Train Shuffled", train[0])
        if args.contrastive:
            train_dataset = Dataset.from_list(train)
            train_dataset.push_to_hub(args.hf_dataset_name,split="train")
        else:
            train_df=pd.DataFrame(train)
            train_df=train_df.drop(columns=["nli_score","nli_logit","nli_accuracy","align_score","align_accuracy","qwen4Bnoth_score","qwen4Bth_score","qwen4Bth_accuracy","qwen8B_score","qwen8B_accuracy","qwen30B_score","qwen30B_accuracy","qwen4Bnoth_accuracy"]) #align_accuracy qwen4Bth_accuracy
            train_df["webpage_references"]=train_df["webpage_references"].apply(lambda x : [''] if not len(x) else x)
            if args.filter_error_type == "qwen_optim":
                subset=["claim_negation","claim_erronous_change","claim_numerical_mismatch","modify_passage-add_relevant_to_claim","claim_combine_facts","claim_add_to_the_claim_contradicting_info","claim_over_infer_claim",'']
            elif args.filter_error_type == "true_optim":
                subset=["claim_erronous_change","claim_numerical_mismatch","modify_passage-add_relevant_to_claim","claim_combine_facts","claim_add_to_the_claim_contradicting_info","modify_passage-add_contradiction","modify_passage-add_conflicting_sources","claim_infer_claim","claim_over_infer_claim",'']
            elif args.filter_error_type == "all_optim":
                subset=["claim_negation","claim_erronous_change","claim_numerical_mismatch","modify_passage-add_relevant_to_claim","claim_combine_facts","claim_add_to_the_claim_contradicting_info","claim_over_infer_claim","modify_passage-add_contradiction","modify_passage-add_conflicting_sources","claim_infer_claim","claim_over_infer_claim",'']
            if args.filter_error_type is not None and len(subset):
                print("Keeping errors:", subset)
                train_df=train_df[train_df["error_type"].isin(subset)]
                print("Hard positive/others:",len(train_df[train_df["example_type"]!="generated"]))

            print(train_df.columns)
            train_dataset = Dataset.from_pandas(train_df)
            train_dataset.push_to_hub(args.hf_dataset_name,split="train")
    if args.dev_file:
        with open(args.dev_file, 'r') as file:
            dev = json.load(file)
            if isinstance(train, dict) and "data" in train.keys():
                train=train["data"]
        dev_dataset = Dataset.from_pandas(pd.DataFrame(pd.DataFrame(dev)))
        dev_dataset.push_to_hub(args.hf_dataset_name,split="dev")
    elif args.og_dev:
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
        if args.subset=="subset_balanced":
            data_path=os.environ['WORK']+ "/AttributionBench"
        if args.subset=="full_data":
            data_path=os.environ['WORK']+ "/AttributionBenchfull_data"
        dev = datasets.load_from_disk(data_path)
        dev_dataset = dev["dev"] #, split=args.split, features=features)
        error_type = [''] * len(dev_dataset)
        example_type = ["other"] * len(dev_dataset)
        og_label = [''] * len(dev_dataset)
        label_switch = ['']* len(dev_dataset)
        dev_dataset = dev_dataset.add_column("error_type", error_type)
        dev_dataset = dev_dataset.add_column("example_type", example_type)
        dev_dataset = dev_dataset.add_column("original_label", og_label)
        dev_dataset = dev_dataset.add_column("label_switch", label_switch)
        dev_dataset.push_to_hub(args.hf_dataset_name,split="dev")
    


    # final_dataset = DatasetDict({
    #     "train": train_dataset,
    #     "dev": dev_dataset
    # })

    
    

    logger.info(f"Dataset Pushed to HF under name {args.hf_dataset_name}")
    if args.save_local:
        logger.info(f"Saving Dataset locally")
        dataset = load_dataset("hanane/"+args.hf_dataset_name, download_mode='force_redownload') 
        dataset.save_to_disk(os.environ['WORK']+ "/"+ args.hf_dataset_name) 
        logger.info(f"Dataset saved to {os.environ['WORK']+ '/'+ args.hf_dataset_name}")