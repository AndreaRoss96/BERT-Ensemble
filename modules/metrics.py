import pandas as pd
import numpy as np



def f1_score(df):
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
        precision = same / pred
        recall = same / ans
        f1.append((2 * precision * recall) / (precision + recall))

    return f1# np.mean(f1)

def exact_match(df):
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
