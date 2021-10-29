import warnings
warnings.filterwarnings('ignore')
import json
import pandas as pd


def create_df(path_to_json, errors = []):
    ''' 
    Parses the Json File and create a DataFrame with "title" - "context" - "question" - "id" - "answer_text" - "idx_start"
    params:
    - path_to_json: path to the dataset file. It needs to be in a json format
    - errors: list of dataset's ids that contain errors and that wont be considered
    returns:
    - dataFrame object containing the dataset 
    '''
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
    '''
    Print to console the formatted prediction with relative informations
    params: 
    - id: answer id
    - record: a dataframe/dictionary with columns/keys: ["pred_start", "pred_end", "context", "offsets"]
    - question: a string containing the question
    - answer: a string containing the true answer as reported in the dataset
    '''
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


def get_errors(path = ""):
    '''opens the file contatining the errors and returns a list of indexes '''
    with open(path, "r") as err:
        e = err.readlines()
    return [el.strip() for el in e]

def index_to_text(record):
    '''
    Transform the token index in the corresponding string in the context:
    params: 
    - record: a dataframe/dictionary with columns/keys: ["pred_start", "pred_end", "context", "offsets"]
    returns: 
    - String containing the referred part of context
    '''
    start_token = record["pred_start"]
    end_token = record["pred_end"]
    context = record["context"]
    offsets = record["offsets"]
    start_char = offsets[start_token][0]
    end_char = offsets[end_token][1]
    return context[start_char:end_char]

def write_prediction(df, output_file):
    '''
    Write the prediction on files 
    params: a dataframe/dictionary with columns/keys: ["pred_start", "pred_end", "context", "offsets"]
    - 
    '''
    d = {}
    for i, row in df.iterrows():
        d[i] = index_to_text(row)
    
    f = open(output_file, "w")
    json.dump(d, f)
    f.close()