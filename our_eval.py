import json
from modules.utils import *
from modules.metrics import *


if __name__ == "__main__":
    path2="D:\\Uni\\2020-21 1 sem\\NLP\proj\\repo\BERT-ensemble\\answer.json"
    path_to_json = "D:\\Uni\\2020-21 1 sem\\NLP\\proj\\repo\\BERT-ensemble\\output_0.json"
    with open(path_to_json) as f:
        pred = json.load(f)
    true = create_df_train(path2, [])
    true.set_index("id", inplace=True)


     
    data = {"pred_text": pd.Series(pred),
        "true_text":true["answer_text"]
        }
    df3 = pd.DataFrame.from_dict(data)
    print(df3)

    f1 = f1_score_string(df3)
    em = exact_match_string(df3)
    print(np.mean(f1))
    print("\n")
    print("="*40)
    print("\n")
    print(em)
    '''
    params:
    - df: a dataframe with the columns: "pred_start", 
                                        "pred_end",
                                        "true_start",
                                        "true_end".
    returns:
    - average f1 score
    print(true)
    '''