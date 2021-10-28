import warnings
warnings.filterwarnings('ignore')
import os
import re
import json
import string
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from modules.dataframe_tools import *


def create_df(path_to_json, errors):
    # Parse the Json File and create a DataFrame with "title" - "context" - "question" - "id" - "answer_text" - "idx_start"
    with open(path_to_json) as f:
        json_txt = json.load(f)

    processed = []
    for item in json_txt["data"]:
        title = item["title"]
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa['id'] not in errors:
                    record_id = qa['id']
                    question = qa["question"]
                    answer_text = qa["answers"][0]["text"]
                    all_answers = [_["text"] for _ in qa["answers"]]
                    start_char_idx = qa["answers"][0]["answer_start"]
                    samples = [record_id, title, context, question, answer_text, start_char_idx]
                    processed.append(samples)

    columns = ["id", "title", "context", "question", "answer_text", "start_idx"]
    df = pd.DataFrame(processed, columns=columns)
    return df

def print_prediction(id, record, question, true_answer):
    # TODO: check if the next line is still needed

    start = record["pred_start"]
    end = record["pred_end"]
    context = record["context"]

    print("id:\t\t{} \n".format(id))
    print("Question: \t{} \n".format(question))
    print("Paragraph:\t{}\n".format(context))
    print("True Answer:\t{}".format(true_answer))
    print("Possib Answer:\t{}".format(index_to_text(record)))
    print("Start: {}, \t end: {}".format(start, end))
    print("==============================================================\n\n")


def get_errors(path = "/content/drive/MyDrive/NLP/proj finale/utils/errors.txt"):
    with open(path, "r") as err:
        e = err.readlines()
    return [el.strip() for el in e]

def index_to_text(record):
    start_token = record["pred_start"]
    end_token = record["pred_end"]
    context = record["context"]
    offsets = record["offsets"]
    start_char = offsets[start_token][0]
    end_char = offsets[end_token][1]
    return context[start_char:end_char]

def write_prediction(df, output_file):
    d = {}
    for i, row in df.iterrows():
        d[i] = index_to_text(row)
    
    f = open(output_file, "w")
    json.dump(d, f)
    f.close()