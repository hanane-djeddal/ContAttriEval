from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datasets

import re
import os
import sys
import json
import argparse
import numpy as np

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(ROOT_PATH)

from tools.citation_tools import reposition_period_after_citation,get_source_from_text,_run_nli_autoais,init_tokenizers, split_text_by_citations


from tqdm import tqdm

def analyze_relevance(claim, long_context, model_name='castorini/monot5-base-msmarco', chunk_size=100, overlap=20):
    """
    Analyzes the relevance of a long context to a claim by splitting it into chunks
    and scoring each chunk with a cross-encoder model.

    Args:
        claim (str): The main claim to be evaluated.
        long_context (str): The long text context.
        model_name (str): The name of the pre-trained cross-encoder model.
        chunk_size (int): The number of words in each chunk.
        overlap (int): The number of overlapping words between chunks.

    Returns:
        dict: A dictionary containing the relevance metrics.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Split the long context into chunks with overlap
    words = long_context.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    if not chunks:
        return None

    relevance_scores = []
    
    # Process each chunk to get a relevance score
    for chunk in chunks:
        # The query and document are tokenized together for the cross-encoder
        input_ids = tokenizer.encode(f"Query: {claim} Document: {chunk}", return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids)
            # The model outputs logits; we're interested in the logit for the 'true' token
            logits = outputs.logits.squeeze()
            true_logit = logits[1].item()  # Assuming true is at index 1

            relevance_scores.append(true_logit)

    # --- Quantify the metrics ---
    import numpy as np

    if not relevance_scores:
        return None

    # Peak Relevance Score
    peak_relevance = np.max(relevance_scores)

    # Number of High-Scoring Chunks (adjust threshold as needed)
    threshold = np.mean(relevance_scores) + np.std(relevance_scores)
    high_scoring_chunks_count = sum(1 for score in relevance_scores if score > threshold)

    # Relevance Variance
    relevance_variance = np.var(relevance_scores)

    # Location of Peak Relevance
    peak_location = np.argmax(relevance_scores)

    return {
        'relevance_scores_list': relevance_scores,
        'peak_relevance': peak_relevance,
        'high_scoring_chunks_count': high_scoring_chunks_count,
        'relevance_variance': relevance_variance,
        'peak_location': peak_location
    }

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser()
    

    parser.add_argument(
        "--file", type=str, help="Path to the dataset file", default=None
    )
    parser.add_argument(
        "--validating_code", action="store_true",  help="validate_code"
    )

    args = parser.parse_args()

    with open(args.file, 'r') as f:
        dataset = json.load(f)

    for idx, row in enumerate(tqdm(dataset)):
        if args.validating_code and  idx == 2:
             break
        metrics = analyze_relevance(row["claim"], " ".join(row["references"]))
        row["relevance_metrics"]=metrics
    code_validation="validating_code_" if args.validating_code else ''

    # results_path = args.file.split('/')
    # results_folder= results_path[0]
    # results_file = results_path[-1] if results_file[-1] !="" else results_file[-2]
    # code_validation="_validating_code" if args.validating_code else ''
    results_file =  args.file[:-5]+"_relevance_in_context_analysis_"+code_validation+".json" #+results_file.split('.')[-2]


    #results_file =  "relevance_in_context_analysis_"+code_validation+args.file
    #print(dataset[0])
    with open(results_file, "w") as f:
        json.dump(dataset, f, indent=4,cls=NpEncoder)

main()


# # --- Example Usage ---
# # Replace with your actual data
# claim_text = "The first person to walk on the moon was Neil Armstrong."

# long_context_text = """
# The Apollo 11 mission launched from Florida on July 16, 1969. The mission's crew consisted of Commander Neil Armstrong, lunar module pilot Buzz Aldrin, and command module pilot Michael Collins. After a four-day journey, they entered lunar orbit. On July 20, the lunar module "Eagle" separated and began its descent. While preparing for the mission, NASA had considered various landing sites. It's a little-known fact that a similar, uncrewed mission was launched a year prior. However, the Apollo 11 mission was a resounding success. Neil Armstrong became the first human to step on the lunar surface, uttering the famous words, "That's one small step for a man, one giant leap for mankind." He was followed a short time later by Buzz Aldrin.
# """

# # Run the analysis
# metrics = analyze_relevance(claim_text, long_