import json
from modules.utils import create_df
if __name__ == "__main__":
    path_to_json = "D:\\Uni\\2020-21 1 sem\\NLP\\proj\\repo\\BERT-ensemble\\dataset\\training_set.json"
    with open(path_to_json) as f:
        json_txt = json.load(f)

    with open("answer.txt", "w") as f:
        json.dump(json_txt["data"][-1], f)
        



df = create_df("answer.json")
print(df)
        


