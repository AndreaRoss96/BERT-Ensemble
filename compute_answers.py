import argparse
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

from keras.models import load_model
from modules.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='given a json file formatted as \
            the training set, creates a prediction  \
            file in the desired format.'
    )
    parser.add_argument('path_to_json', metavar='data.json', help='Path to json testing file')
    parser.add_argument('--model',      default='ensemble', type= str, help='type of model you want to use to compute the answers: [ensemble, vanilla]')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='BERT tokenizer model')
    parser.add_argument('--max_len',    default=512, type=int, help='maximum len for the BERT tokenizer model')
    args = parser.parse_args()
    path_to_json = args.path_to_json

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    with open(path_to_json) as f:
        raw_test_data = json.load(f)
    
    df_orig = create_df(raw_test_data, [])

    df = process_dataset(df_orig, tokenizer, answer_available=True, max_len=args.max_len)
    x_test, y_test = dataframe_to_array(df)

    