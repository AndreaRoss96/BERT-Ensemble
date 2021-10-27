import argparse
import json
from typing import DefaultDict
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from transformers import AutoTokenizer

from modules.dataframe_tools import *
from modules.utils import *
from modules.BERTmodels import *

import time

def get_models(add_layer, avg_layer, max_layer, min_layer, mul_layer, sub_layer, max_len, saved_models_path, inputs):
    bert_cnn = create_bert_CNN(inputs=inputs)
    bert_cnn.load_weights(saved_models_path + "BERT_1DConv.hdf5")

    models = [bert_cnn]

    if str(add_layer).lower() =='true':
        bert_add = create_bert_custom(custom_layer="add", inputs=inputs)
        bert_add.load_weights(saved_models_path + "bert-base-uncased_add.hdf5")
        models.append(bert_add)
    if str(avg_layer).lower() =='true':
        bert_avg = create_bert_custom(custom_layer="average", inputs=inputs)
        bert_avg.load_weights(saved_models_path + "bert-base-uncased_average.hdf5")
        models.append(bert_avg)
    if str(max_layer).lower() =='true':
        bert_max = create_bert_custom(custom_layer="maximum", inputs=inputs)
        bert_max.load_weights(saved_models_path + "bert-base-uncased_maximum.hdf5")
        models.append(bert_max)
    if str(min_layer).lower() =='true':
        bert_min = create_bert_custom(custom_layer="minimum", inputs=inputs)
        bert_min.load_weights(saved_models_path + "bert-base-uncased_minimum.hdf5")
        models.append(bert_min)
    if str(mul_layer).lower() =='true':
        bert_mul = create_bert_custom(custom_layer="multiply", inputs=inputs)
        bert_mul.load_weights(saved_models_path + "bert-base-uncased_multiply.hdf5")
        models.append(bert_mul)
    if str(sub_layer).lower() =='true':
        bert_sub = create_bert_custom(custom_layer="subtract", inputs=inputs)
        bert_sub.load_weights(saved_models_path + "bert-base-uncased_subtract.hdf5")
        models.append(bert_sub)

    return models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='given a json file formatted as \
            the training set, creates a prediction  \
            file in the desired format.'
    )
    parser.add_argument('path_to_json',         metavar='data.json', help='Path to json testing file')
    parser.add_argument('--model',              default='ensemble', type= str, help='type of model you want to use to compute the answers: [ensemble, vanilla]')
    parser.add_argument('--bert_model',         default='bert-base-uncased', type=str, help='BERT tokenizer model')
    parser.add_argument('--max_len',            default=512,    type=int, help='maximum len for the BERT tokenizer model')
    parser.add_argument('--add_layer',          default='true', type=str, help='use an BERT model with an add layer for the ensemble')
    parser.add_argument('--avg_layer',          default='true', type=str, help='use an BERT model with an avg layer for the ensemble')
    parser.add_argument('--max_layer',          default='true', type=str, help='use an BERT model with a max layer for the ensemble')
    parser.add_argument('--min_layer',          default='true', type=str, help='use an BERT model with a min layer for the ensemble')
    parser.add_argument('--mul_layer',          default='true', type=str, help='use an BERT model with a mul layer for the ensemble')
    parser.add_argument('--sub_layer',          default='true', type=str, help='use an BERT model with a sub layer for the ensemble')
    parser.add_argument('--saved_model_path',   default='saved_models/', type=str, help='path with the models weight saved')
    parser.add_argument('--output_json',        default='output_0.json', type= str, help='Folder with the predicted values')

    args = parser.parse_args()
    path_to_json = args.path_to_json
    max_len = args.max_len
    saved_models_path = args.saved_model_path

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Create Data set for testing    
    df_orig = create_df(path_to_json, [])

    #TODO remove this vvvv
    df_orig = df_orig[:10] # <<
    #TODO ^^^^^^^^^^^^^^^^
    print("\nProcessing dataset ...\n")
    df = process_dataset(df_orig, tokenizer, answer_available=False, max_len=max_len)
    x_test = dataframe_to_array(df, answer_available = False)
    print("\nProceding to create models ...\n")

    # Create input layer for the neural networks
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name="token_type_ids")
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    inputs = [input_ids, token_type_ids, attention_mask]

    # Create vanilla Bert model
    bert_van = create_bert_vanilla(inputs=inputs)
    bert_van.load_weights(saved_models_path + "BERT_Vanilla.hdf5")

    if args.model == 'ensemble' :
        # if ensemble selected (default) --> create enemble model adding vanilla bert model
        models = get_models(
            args.add_layer,
            args.avg_layer,
            args.max_layer,
            args.min_layer,
            args.mul_layer,
            args.sub_layer,
            max_len=max_len,
            saved_models_path = saved_models_path,
            inputs=inputs
        )
        models.append(bert_van)
        model = EnsembleModel(models, inputs)
    elif args.model == 'vanilla' : 
        model = bert_van
    else :
        raise ValueError("Check \'--model\' value: use \'endemble\' or \'vanilla\'")
    
    print("\nPrediction in process ...\n")
    t0 = time.time() 
    pred = model.predict(x_test)
    t1 = time.time()
    print(f"the prediction process took {t1-t0} seconds")  
    print()  

    data = {
        "id" : df_orig.id.values,
        "pred_start":pred[0],
        "pred_end": pred[1],
        "input_ids": df["input_ids"],
        "context" : df_orig["context"].values,
        "offsets": df["offsets"].values}

    df_res = pd.DataFrame.from_dict(data)
    df_res.set_index("id", inplace=True)

    write_prediction(df_res, args.output_json)
    


