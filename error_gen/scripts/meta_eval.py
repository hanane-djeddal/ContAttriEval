import os
import json
import re
import argparse
import logging
from tabulate import tabulate
import pandas as pd
import sys
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

from sklearn import metrics

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.append(ROOT_PATH)

from tools.eval_tools import read_json_files_from_folder, compute_metrics


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




def run_concat_files():
    parser = argparse.ArgumentParser()    
    
    parser.add_argument(
        "--concat_files", action="store_true", help="concatenate files"
    )
    
    parser.add_argument(
        "--folder", type=str, default=None, help="Folder containing list of files"
    )
    parser.add_argument(
        "--results_file", type=str, default=None, help="Name of results file"
    )
    parser.add_argument(
        "--eval_file", type=str, default=None, help="Name of results file"
    )
    
    parser.add_argument(
        "--evaluate", type=str, default=None, help="Evaluate results",choices=["acc", "f1","auc-roc"]
    )
    parser.add_argument(
        "--scoredlabels", action="store_true", help="prediction is a score"
    )
    
    parser.add_argument(
        "--without_file_update", action="store_true", help="update file"
    )    
    
    parser.add_argument(
        "--group_by_column",type=str, default="src_dataset", help=" group_by_column"
    )
    
    parser.add_argument(
        "--prediction_column",type=str, default="auto_score", help="predictin column"
    )

    parser.add_argument(
        "--prediction_score_column",type=str, default="logit", help="predictin column"
    )

    parser.add_argument(
        "--score_column",type=str, default="logit", help="score column"
    )
    
    parser.add_argument(
        "--alignscore", action="store_true", help="update file"
    )    

    args = parser.parse_args()

    #################
    # Logging params
    #################
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)} - {parser.get_default(arg)}")
    if args.concat_files and args.folder:
        data = read_json_files_from_folder(args.folder)
    elif args.eval_file:
        with open(args.eval_file, 'r') as file:
            data = json.load(file)
            if isinstance(data, list):
                data ={"data":data}
    else:
        logger.info("Please Provide result file to evaluate or folder")
        return
    if args.evaluate == "acc":

        
        all_scores=compute_metrics(data["data"],prediction_column=args.prediction_column,scoredlabels=args.scoredlabels,group_by_column=args.group_by_column,score_column=args.score_column,alignscore=args.alignscore)
        data["evaluation2"]=all_scores[0].to_dict('index')
        data["evaluation3"]=all_scores[1].to_dict('index')

    if not args.without_file_update:
        if args.eval_file:
            results_file= args.eval_file
        else:
            results_file = os.path.join(args.folder, args.results_file)
        with open(results_file, "w") as f:
            json.dump(data, f, indent=4)

run_concat_files()
