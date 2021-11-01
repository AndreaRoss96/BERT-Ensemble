import warnings
warnings.filterwarnings('ignore')
import os
import re
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
#from transformers import AutoTokenizer, TFAutoModel
from modules.utils import *


def print_error_train(id, question, context, answer, start_idx, end_idx):
    '''
    Generic error print for a train dataset
    params:
    - id: record id.
    - question:     record's question.
    - context:      record's context.
    - answer:       record's answer.
    - start_idx:    computed start index of true answer.
    - end_idx:      computed end index of true answer.
    '''
    print("id:\t\t{} \n".format(id))
    print("Question: \t{} \n".format(question))
    print("Paragraph:\t{}\n".format(context))
    print("True Answer:\t{}".format(answer))
    print("Possib Answer:\t{}".format(context[start_idx:end_idx]))
    print("Bounds: {} {} ---- Length: {}\n\n".format(start_idx, end_idx, len(context)))
    print("==============================================================\n\n\n\n")

def print_error_test(id, question, context):
    """
    Generic error print for a test dataset
    params:
    - id: record id.
    - question:     record's question.
    - context:      record's context.
    """
    print("id:\t\t{} \n".format(id))
    print("Question: \t{} \n".format(question))
    print("Paragraph:\t{}\n".format(context))
    print("==============================================================\n\n")


def process_test_record(record, tokenizer, max_len = 512, show_bast = False ):
    '''
    This function tokenizes the train record (without answers) and returns a list with the information needed for a Bert model.
    
    Params:
    - record:       a single row of the dataset. Here the record has the form: id, title, context, question
    - tokenizer:    the specific tokenizer it needs to be utilized
    - max_len:      maximum length accepted by the BERT model.
    - show_bast:    debugging option, shows when computed data are longer than max_len
    Returns:
    - [id, input_ids, attention_mask, token_type_ids, offset]: if the computation is successfull
    - ["","","","","",""]: if the computation went wrong
    '''
    id = record["id"]
    title = record["title"]
    context = record["context"]
    question = record["question"]

    error_return = ["","","","","",""]


    # Clean context, answer and question from unnecessary whitespaces
    context = " ".join(str(context).split())
    question = " ".join(str(question).split())

    # Tokenize context and question
    tokenized_context = tokenizer(context, return_offsets_mapping=True)
    tokenized_question = tokenizer(question, return_offsets_mapping=True)
    offsets = tokenized_context.offset_mapping

    # Find start and end token index for tokens from answer
    max_ctx_space = max_len - len(tokenized_question.input_ids)
    interval = [0,max_ctx_space]
    '''
    # change questions where input_ids would be > max_len
    if  len(tokenized_question.input_ids) + len(tokenized_context.input_ids) > max_len:
        # truncate the context at max_ctx_space
        interval = [0, max_ctx_space]
    '''

    # Create inputs take [CLS] and [SEP] from question
    input_ids = tokenized_context.input_ids[interval[0]:interval[1]] + tokenized_question.input_ids[1:]
    token_type_ids = [0] * len(tokenized_context.input_ids[interval[0]:interval[1]]) + [1] * len(tokenized_question.input_ids )
    attention_mask = [1] * len(input_ids)

    
    # Pad and create attention masks.
    # Skip if truncation is needed
    padding_length = max_len - len(input_ids)
    if padding_length > 0:  # pad
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

    elif padding_length < 0: 
        if show_bast:

            print("Invalid question: max_lenght reached")
            print_error_test(id, question, context)        

        return error_return

    return [id, title, input_ids, attention_mask, token_type_ids, offsets]


def process_train_record(record, tokenizer, max_len = 512, show_oub = False, show_bast = False, show_no_answ = False ):
    '''
    This function tokenizes the train record (complete with answers) and returns a list with the information needed for a Bert model.
    Params:
    - record:           a single row of the dataset. Here the record has the form: id, title, context, question, answer_text, start_idx
    - tokenizer:        the specific tokenizer it needs to be utilized
    - max_len:          maximum length accepted by the BERT model.
    - show_oub:         debugging option, shows when true answer is out of bound wrt the context
    - show_bast:        debugging option, shows when computed data are longer than max_len
    - show_no_answ:     debugging option, shows when the tokenized answer is empty
    Returns:
    - [id, input_ids, attention_mask, token_type_ids, start_token_idx, end_token_idx, offset]: if the computation is successfull
    - ["","","","","","",""]: if the computation went wrong
    '''
    id = record["id"]
    title = record["title"]
    context = record["context"]
    question = record["question"]
    answer = record["answer_text"]
    start_idx = record["start_idx"]

    error_return = ["","","","","","","",""]


    # Clean context, answer and question from unnecessary whitespaces
    context = " ".join(str(context).split())
    question = " ".join(str(question).split())
    answer = " ".join(str(answer).split())

    # Find end character index of answer in context
    end_idx = start_idx + len(answer)
    # Find errors in the dataset
    if end_idx > len(context):
        if show_oub:
            print("Invalid question: out of bound answer")
            print_error_train(id, question, context, answer, start_idx, end_idx)        
        return  error_return

    # Mark the character indexes in context that are in answer
    is_char_in_ans = [0] * len(context)
    for idx in range(start_idx, end_idx):
        is_char_in_ans[idx] = 1

    # Tokenize context and question
    tokenized_context = tokenizer(context, return_offsets_mapping=True)
    tokenized_question = tokenizer(question, return_offsets_mapping=True)

    # Find tokens that were created from answer characters
    offsets = tokenized_context.offset_mapping
    ans_token_idx = []
    for idx, (start, end) in enumerate(offsets):
        if sum(is_char_in_ans[start:end]) > 0:
            ans_token_idx.append(idx)

    if len(ans_token_idx) == 0:
        if show_no_answ:
            print("Invalid question: no answer token")
            print_error_train(id, question, context, answer, start_idx, end_idx)        
        return error_return

    # Find start and end token index for tokens from answer
    start_token_idx = ans_token_idx[0]
    end_token_idx = ans_token_idx[-1]
    max_ctx_space = max_len - len(tokenized_question.input_ids)
    interval = [1,max_ctx_space]

    # change questions where input_ids would be > max_len
    if  len(tokenized_question.input_ids) + len(tokenized_context.input_ids) > max_len:
        
        # Consider only the context part that has more influence on the answer 
        answer_len = end_token_idx - start_token_idx
        remain_space = max_len - len(tokenized_question.input_ids) - answer_len
        interval = [start_token_idx - remain_space/2, end_token_idx + remain_space/2]

        # if the proposed interval is out of bound wrt the context, the interval is
        # changed accordingly
        
        # Note that the two if statement cannot be true at the same time
        if start_token_idx - remain_space/2 < 0:
            interval = [1, end_token_idx + remain_space - start_token_idx]
        if start_token_idx + remain_space/2 > max_ctx_space :
            interval = [max_ctx_space - answer_len - remain_space, max_ctx_space ]

    # Create inputs take [CLS] and [SEP] from context
  
    input_ids = [101] + tokenized_context.input_ids[interval[0]:interval[1]]  + tokenized_question.input_ids[1:]
    token_type_ids = [0] * len(tokenized_context.input_ids[interval[0]:interval[1]]) + [1] * len(tokenized_question.input_ids )
    attention_mask = [1] * len(input_ids)

    
    # Pad and create attention masks.
    # Skip if truncation is needed
    padding_length = max_len - len(input_ids)
    if padding_length > 0:  
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

    elif padding_length < 0:  
        if show_bast:
            print("Contex + answer too long")
            print_error_train(id, question, context, answer, start_idx, end_idx)        
        return error_return
    
    return [id, title, input_ids, attention_mask, token_type_ids, start_token_idx, end_token_idx, offsets]



def process_dataset(df, tokenizer, answer_available=True, max_len=512):
    '''
    Function that processes the whole dataset changing the proceduere based on the presence of answers in the dataset
    params:
    - df: the dataframe
    - tokenizer: the tokenizer specific for the bert model used
    - answer_available: True if the answer is in the dataset False if it is not
    - max_len: maximum length accepted by the dataset
    returns:
    - if answer_available=True a dataframe with columns    ["id" (index),
                                                            "title", 
                                                            "input_ids", 
                                                            "attention_mask", 
                                                            "token_type_ids", 
                                                            "start_token_idx", 
                                                            "end_token_idx", 
                                                            "offsets"] 
    - if answer_available=True a dataframe with columns    ["id" (index),
                                                            "title", 
                                                            "input_ids", 
                                                            "attention_mask", 
                                                            "token_type_ids", 
                                                            "offsets"]
    
    '''
    if answer_available:
        tmp = [process_train_record(record, tokenizer, max_len) for _, record in df.iterrows()]    
        columns = ["id","title", "input_ids", "attention_mask", "token_type_ids", "start_token_idx", "end_token_idx", "offsets"]
    else:
        tmp = [process_test_record(record, tokenizer, max_len) for _, record in df.iterrows()]    
        columns = ["id","title", "input_ids", "attention_mask", "token_type_ids", "offsets"]

    proc_df = pd.DataFrame(tmp, columns=columns).set_index(["id"])

    proc_df.replace("", float("NaN"), inplace=True)
    proc_df.dropna(inplace=True)

    return proc_df
    
def train_test_split_on_title(df, split_val=0.75):
    """
    From the process dataset the function splits it into train and validation, wrt title.

    And then prepares these dataset for the model fit functions

    Return:
     - train_x: "input_ids", "token_type_ids", "attention_mask"
     - train_y: "start_token_idx", "end_token_idx"
     - val_x:   "input_ids", "token_type_ids", "attention_mask"
     - val_y:   "start_token_idx", "end_token_idx"
    """
    r_df = df.reset_index()

    #split based on the title
    split_title = r_df['title'].iloc[int(split_val * len(r_df)) - 1]
    split_index = r_df[r_df['title'] == split_title].index.min()
    df_train, df_val = r_df.iloc[:split_index], r_df.iloc[split_index:]

    # creating test and val x,y
    train_x, train_y = dataframe_to_array(df_train)
    val_x, val_y = dataframe_to_array(df_val)
    return train_x, train_y, val_x, val_y

def dataframe_to_array(df, answer_available = True):
    """
    Transforms a Dataframe into a lists of np.array
    params:
    - df:   dataframe with the form    ["input_ids", 
                                        "token_type_ids", 
                                        "attention_mask", 
                                        "start_token_idx", 
                                        "end_token_idx"] 
            if answe_available=True
            
            dataframe with the form    ["input_ids", 
                                        "token_type_ids", 
                                        "attention_mask"] 
            if answe_available=True
    - answer_available: True if the answer is in the dataset False if it is not

    returns:
    - df_x:     list containing the columns "input_ids", "token_type_ids", "attention_mask", as np.array
    - df_y:     if answer_available = True. List containing the columns "start_token_idx", "end_token_idx"
    """
    df_list = df.to_dict(orient='list')
    x_set = ["input_ids", "token_type_ids", "attention_mask"]
    if answer_available:
        y_set = ["start_token_idx", "end_token_idx"]
        df_x = [np.array(df_list[x]) for x in x_set]
        df_y = [np.array(df_list[y]) for y in y_set]
        return df_x, df_y
    else:
        df_x = [np.array(df_list[x]) for x in x_set]
        return df_x


