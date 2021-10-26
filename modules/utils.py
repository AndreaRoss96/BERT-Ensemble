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


def create_df(json_txt, errors):
    # Parse the Json File and create a DataFrame with "title" - "context" - "question" - "id" - "answer_text" - "idx_start"
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

def print_prediction(p, df_orig):
    # TODO: check if the next line is still needed
    first_idx = df_orig.first_valid_index()
    for i, row in enumerate(zip(p[0],p[1])):

        start = row[0]
        end = row[1]
        id = df_orig["id"][i+first_idx]
        context = df_orig["context"][i+first_idx]
        question = df_orig["question"][i+first_idx]
        answer = df_orig["answer_text"][i+first_idx]
        start_true = df["start_token_idx"][i]
        end_true = df["end_token_idx"][i]

        print("id:\t\t{} \n".format(id))
        print("Question: \t{} \n".format(question))
        print("Paragraph:\t{}\n".format(context))
        print("True Answer:\t{}".format(answer))
        print("True Answer brutta:\t{}".format(tokenizer.convert_ids_to_tokens(x[0][i][start_true: end_true+1])))
        print("Possib Answer:\t{}".format(tokenizer.convert_ids_to_tokens(x[0][i][start: end+1])))
        print("Start: {}, \t end: {}".format(start, end))
        print("==============================================================\n\n")

def get_errors(path = "/content/drive/MyDrive/NLP/proj finale/utils/errors.txt"):
    with open(path, "r") as err:
        e = err.readlines()
    return [el.strip() for el in e]

