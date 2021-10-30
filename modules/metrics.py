import pandas as pd
import numpy as np


def f1_score_string(df):
    ''' 
    params:
      - df: a dataframe with the columns: "pred_text", 
                                          "true_text",
    returns:
      - average f1 score

    '''
    f1 = []
    for i, row in df.iterrows():
        pred_text = row["pred_text"].split()
        true_text = row["true_text"].split()
        pred = len(pred_text)
        ans = len(true_text)
        same = len([i for i in pred_text if i in true_text])
        if same == 0:
            f1.append(0)
        else:
            precision = same / pred
            recall = same / ans
            f1.append((2 * precision * recall) / (precision + recall))
    return f1# np.mean(f1)

def f1_score_indexed(df):
    ''' 
    params:
      - df: a dataframe with the columns: "pred_start", 
                                          "pred_end",
                                          "true_start",
                                          "true_end".
    returns:
      - average f1 score

    '''
    f1 = []
    for i, row in df.iterrows():
        a = row["pred_start"] if row["pred_start"]>row["true_start"] else row["true_start"]
        b = row["pred_end"] if row["pred_end"]<row["true_end"] else row["true_end"]
        pred = row["pred_end"] - row["pred_start"]  +1
        ans = row["true_end"] - row["true_start"] +1
        same = max(0, b-a+1)
        if same == 0:
            f1.append(0)
        else:
            precision = same / pred
            recall = same / ans
            f1.append((2 * precision * recall) / (precision + recall))
    return f1# np.mean(f1)

def exact_match_string(df):
    ''' 
    params:
      - df: a dataframe with the columns: "pred_text", 
                                          "true_text",
    returns:
      - rate of exact matches
    '''

    count = 0
    for i, row in df.iterrows():
        if row["pred_text"] == row["true_text"] :
            count -=- 1
    return count / df.shape[0]

def exact_match_indexed(df):
    ''' 
    params:
      - df: a dataframe with the columns: "pred_start", 
                                          "pred_end",
                                          "true_start",
                                          "true_end".
    returns:
      - rate of exact matches
    '''

    count = 0
    for i, row in df.iterrows():
        if row["pred_start"] == row["true_start"] and row["pred_end"] == row["true_end"]:
            count -=- 1
    return count / df.shape[0]
