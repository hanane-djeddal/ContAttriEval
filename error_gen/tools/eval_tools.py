import os
import json
import re
import argparse
import logging
from tabulate import tabulate
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import glob
import numpy as np

from sklearn import metrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def concatenate_csv_files(folder_path):
    """
    Reads all CSV files in a folder, adds a 'data_source' column
    with the filename (without extension), and concatenates them
    into a single pandas DataFrame.

    Args:
        folder_path (str): The path to the folder containing the CSV files.

    Returns:
        pandas.DataFrame: The combined dataset, or None if no CSV files are found.
    """
    # 1. Find all files ending with '.csv' in the specified folder
    # glob.glob returns a list of file paths
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not all_files:
        logger.info(f"No CSV files found in the folder: {folder_path}")
        return None
    else:
        logger.info(f"Found the following data files: {all_files}")

    data_frames = []

    # 2. Loop through each file
    for file_path in all_files:
        # Extract the filename without the path and extension
        # e.g., 'path/to/my_data.csv' -> 'my_data'
        filename_with_ext = os.path.basename(file_path)
        data_source_name = (os.path.splitext(filename_with_ext)[0]).replace("_download",'')

        try:
            # 3. Read the CSV file into a DataFrame
            df = pd.read_csv(file_path,index_col=0)

            # 4. Add the 'data_source' column
            df['src_dataset'] = data_source_name

            # 5. Append the DataFrame to the list
            data_frames.append(df)
            print(f"Successfully processed: {filename_with_ext}")

        except Exception as e:
            print(f"Error reading file {filename_with_ext}: {e}")
            continue

    # 6. Concatenate all DataFrames in the list
    combined_df = pd.concat(data_frames, ignore_index=True)

    print("\nConcatenation complete!")
    return combined_df


def read_json_files_from_folder(folder_path):
    pattern = re.compile(r'(\d+)-(\d+)')
    pattern_start = re.compile(r'^(\d+)-')
    file_pattern = re.compile(r'^(\d+)-(\d+)(.+)\.json$')

    json_files = [f for f in os.listdir(folder_path)  if file_pattern.match(f) and f.endswith('.json')]
    logger.info(f"Merging Files {json_files}")
    json_files.sort(key=lambda f: int(pattern_start.match(f).group(1)))
    data = []
    ranges= []
    params = None

    for json_file in json_files:
        matches = pattern.findall(json_file)
        for match in matches:
            ranges.append((int(match[0]), int(match[1])))
            print("iterations:",int(match[0]), int(match[1]))
        file_path = os.path.join(folder_path, json_file)
        with open(file_path, 'r') as file:
            try:
                content = json.load(file)
                if not len(data) or (len(data) and data[-1] != content["data"][0]): #["question"]
                    data.extend(content["data"])
                else:
                    if len(data):
                        print("dublicates",data[-1],content["data"][0])
                    data.extend(content["data"][1:])
                params = {} #content["params"]
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {file_path}")
    logger.info(f"Read {len(data)} lines")
    return {"data":data,"params":params}

def calculate_metrics_per_group(group,gold_column="attribution_label",prediction_column="auto_score",group_by_column="src_dataset", data_len=100):
    """Calculates Accuracy, F1, FP, and FN for a given set of true and predicted labels."""
    print("-----Size of group ",group[group_by_column].unique() ,len(group),"out of ", data_len)
    percentage=len(group)/data_len*100
    print(f'Percentage: {percentage:.2f} %')
    y_true = group[gold_column]
    y_pred = group[prediction_column]

    # Compute Confusion Matrix: [[TN, FP], [FN, TP]]
    # Explicitly setting labels=[0, 1] ensures the order is correct:
    # Row 0: True Negative class (0), Row 1: True Positive class (1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Extract FP and FN
    # cm[0, 1] is the cell for True=0 (Negative) and Pred=1 (Positive) -> False Positive (FP)
    # cm[1, 0] is the cell for True=1 (Positive) and Pred=0 (Negative) -> False Negative (FN)
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]

    return pd.Series({
        'Accuracy': round(accuracy_score(y_true, y_pred)*100,2),
        'F1-Score': round(f1_score(y_true, y_pred)*100,2),
        'False Positives (FP)': round((false_positives/len(y_true)) *100,2),
        'False Negatives (FN)': round((false_negatives/len(y_true)) *100,2),
    })

def compute_auc_roc(group,gold_column="attribution_label",prediction_column="auto_score",group_by_column="src_dataset"):
    y_true = group[gold_column]
    y_score = group[prediction_column]
    roc_auc_mertic=roc_auc_score(y_true=y_true,y_score=y_score)

    fpr, tpr, thresholds =roc_curve(y_true, y_score, pos_label=1)
    return pd.Series({
        'roc_auc_score': roc_auc_mertic,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    })

def compute_metrics(data,gold_column="attribution_label",prediction_column="auto_score", group_by_column="src_dataset", score_column="logit", scoredlabels=False,available_prediction=True,alignscore=False):
    df = pd.DataFrame(data)
    # Map the string true labels to binary labels (1 for the positive class, 0 for the negative class)
    # 'class1' (positive class) -> 1
    # 'class2' (negative class) -> 0
    df[gold_column] = df[gold_column].map({'attributable': 1, 'not attributable': 0})

    if group_by_column=="error_type":
        og_data_len=len(df[df["example_type"]=="hard_positive"])
    else:
        og_data_len=len(df)
    if scoredlabels:
        if alignscore:
            df[score_column] = df[prediction_column] #.apply(lambda x : x)
            df[prediction_column] = df[prediction_column].apply(lambda x : 1 if x>0.6 else 0).astype(int)
        else:   
            df[score_column]=df.apply(lambda x : 1-x[score_column] if x[prediction_column]==0 else x[score_column],axis=1)

            df[group_by_column] = df[group_by_column].replace('', np.nan)
        auc_roc_metrics_per_source = df.groupby(group_by_column).apply(
            compute_auc_roc, 
            gold_column=gold_column, 
            prediction_column=score_column,
        )
        print(auc_roc_metrics_per_source) #.to_markdown(floatfmt=".4f"))
                
        if not available_prediction:
            df[prediction_column] = df[prediction_column].apply(lambda x : 1 if x>0.6 else 0).astype(int)
    else:
        df=df.dropna(subset=[prediction_column])
        auc_roc_metrics_per_source = None

    #df[prediction_column] = df[prediction_column].astype(int)
    metrics_per_source = df.groupby(group_by_column).apply(
        calculate_metrics_per_group, 
        gold_column=gold_column, 
        prediction_column=prediction_column,
        group_by_column=group_by_column,
        data_len=og_data_len,
    )
    print(metrics_per_source.to_markdown(floatfmt=".4f"))
    average_f1 = metrics_per_source['F1-Score'].mean()
    print(f"\n\nAverage F1-Score: {average_f1:.1f}")

    #### printing for easy copy to overleaf
    order_option_1 = ['ExpertQA', 'Stanford-GenSearch', 'AttributedQA', 'LFQA']  # ID
    order_option_2 = ['BEGIN', 'AttrScore-GenSearch', 'HAGRID']  # OOD

    # Get actual sources in the DataFrame
    available_sources = metrics_per_source.index.tolist()

    # Determine which order to use
    if all(source in available_sources for source in order_option_1):
        desired_order = order_option_1
    elif all(source in available_sources for source in order_option_2):
        desired_order = order_option_2
    else:
        # Fallback: use the order as they appear in the DataFrame
        print("Warning: Neither predefined order matches available sources. Using default order.")
        print(f"Available sources: {available_sources}")
        desired_order = available_sources

    # Filter to only include sources that exist (extra safety check)
    desired_order = [source for source in desired_order if source in available_sources]

    # Print individual source values in the specified order
    if desired_order:
        print("---Copying for Overleaf in order:",desired_order)
        metrics_list = []
        for source in desired_order:
            f1 = metrics_per_source.loc[source, 'F1-Score']
            #fp = metrics_per_source.loc[source, 'False Positives (FP)']
            #fn = metrics_per_source.loc[source, 'False Negatives (FN)']
            metrics_list.extend([f'{f1:.1f}']) #, f'{fp:.2f}', f'{fn:.2f}'])
        
        print(f"Metrics (F1 & FP & FN per source): {' & '.join(metrics_list)}")
    else:
        print("Error: No matching sources found in the results.")
   
        

    print("\n\n--- Overall Metrics ---")
    overall_metrics = calculate_metrics_per_group(
        df, 
        gold_column=gold_column, 
        prediction_column=prediction_column
    ).to_frame(name='Overall')
    print(overall_metrics.T.to_markdown(floatfmt=".4f"))

    return overall_metrics, metrics_per_source,auc_roc_metrics_per_source