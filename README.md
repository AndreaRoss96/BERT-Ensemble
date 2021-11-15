# BERT-ensemble
To run compute_answer.py you need to follow these steps:
- Download the model's weights from this [link](https://liveunibo-my.sharepoint.com/:f:/g/personal/filippo_orazi_studio_unibo_it/EqKufBDXOoNEhFs5W1ojlgsBF0074Uesdu6G7t9jhz7Zjw?e=uw6gqG) and save them in the folder _saved_models_ 
    - To run the default model (ensemble VCM) you only need the files called  _bert-base-uncased_vanilla_, _bert-base-uncased_cnn_ and _bert-base-uncased_multiply_
- Run compute answer as 
```
python3 compute_answers.py *path_to_json_file*
```

## Compute answer functionalities
Compute answer has some optional flag:
* ```--path_to_json ``` &rarr; specify the path for the json file.
* ```--bert_model ``` &rarr; specify the BERT model to use. Note that the default model is _bert_based_uncased_ and that the saved weight refer to this implementation.
* ```--model ``` &rarr; specify the model to be run, the possible options are: _vanilla_, _cnn_, _ensemble_. 
* ```--max_len ``` &rarr; specify the max length of the BERT model. Note that the default length is 512 and teh saved weight use this length.
* ```--van_layer ``` &rarr; use a vanilla BERT model for the ensemble.
* ```--cnn_layer ``` &rarr; use a BERT model with a cnn layer for the ensemble.
* ```--add_layer ``` &rarr; use a BERT model with an add layer for the ensemble.
* ```--avg_layer ``` &rarr; use a BERT model with an avg layer for the ensemble.
* ```--max_layer ``` &rarr; use a BERT model with an max layer for the ensemble.
* ```--min_layer ``` &rarr; use a BERT model with a min layer for the ensemble.
* ```--mul_layer ``` &rarr; use a BERT model with a mul layer for the ensemble.
* ```--sub_layer ``` &rarr; use a BERT model with a sub layer for the ensemble.
* ```--saved_model_path ``` &rarr; path to the saved models. Default is _saved_models/_.
* ```--output_json ``` &rarr; Folder with the predicted values and name of the prediction file.

## BERT-ENSEMBLE repository description
The full experiment can be found at this [link](https://github.com/filorazi/BERT-ensemble). The repository follows this structure
- dataset: Folder that contains the dataset
- modules: Folder that contains utility modules:
    - BERTmodels.py: contains the function that returned the models considered
    - dataframe_tools.py: contains the function to process the dataframe
    - utils.py: contains general utility function
- prediction: Folder that contains json files of the prediction obtained during the testing phase
- saved_models: Folder that contains the weights of the models
- answer.json: Testing file with ground truth
- compute_answer.py: script to obtain predictions as described before
- create_answer.py: script to create answer.json
- errors.txt: ids of the errors found in the training set 
- evaluate.py: evaluation script provided
- exploratory_analysis.ipynb: file containing a step by step analysis of the training set provided
- training.ipynb: file containing a step by step training of every model
