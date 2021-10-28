import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFAutoModel



def pred_argmax(prediction):
    """
    prediction : touple of lists
    """
    starts = []
    ends = []
    for start_preds, end_preds in zip(prediction[0],prediction[1]):
      starts.append(np.argmax(start_preds))
      ends.append(np.argmax(end_preds))
    return [np.array(starts),np.array(ends)]

def create_bert_CNN(model_name = 'bert-base-uncased', max_len = 512, inputs=[]):
    ## BERT encoder
    encoder = TFAutoModel.from_pretrained(model_name)

    embedding = encoder(
        input_ids = inputs[0], token_type_ids=inputs[1], attention_mask=inputs[2]
    ).last_hidden_state
    
    start_logits = layers.Conv1D(768, 10, input_shape=embedding.shape[1:], padding="same")(embedding)
    start_logits = layers.Dense(1, use_bias=False, name='start_logit_cnn')(start_logits)
    start_logits = layers.Flatten(name = 'flatten_start_cnn')(start_logits)

    end_logits = layers.Conv1D(768, 10, input_shape=embedding.shape[1:], padding="same")(embedding)
    end_logits = layers.Dense(1, use_bias=False, name='end_logits_cnn')(end_logits)
    end_logits = layers.Flatten(name = 'flatten_end_cnn')(end_logits)

    start_probs = layers.Activation(keras.activations.softmax, name = 'start_pred_cnn')(start_logits)
    end_probs = layers.Activation(keras.activations.softmax, name = 'end_pred_cnn')(end_logits)
   
    embedding.trainable = False

    model = keras.Model(
        inputs=inputs,
        outputs=[start_probs, end_probs],
    )
    return model

def create_bert_vanilla(model_name = 'bert-base-uncased', max_len = 512, inputs=[]):
    ## BERT encoder
    encoder = TFAutoModel.from_pretrained(model_name)

    embedding = encoder(
        input_ids = inputs[0], token_type_ids=inputs[1], attention_mask=inputs[2]
    ).last_hidden_state
    
    start_logits = layers.Dense(1, use_bias=False, name='start_logit_vanilla')(embedding)
    start_logits = layers.Flatten(name = 'flatten_start_vanilla')(start_logits)

    end_logits = layers.Dense(1, use_bias=False, name='end_logits_vanilla')(embedding)
    end_logits = layers.Flatten(name = 'flatten_end_vanilla')(end_logits)

    start_probs = layers.Activation(keras.activations.softmax, name = 'start_pred_vanilla')(start_logits)
    end_probs = layers.Activation(keras.activations.softmax, name = 'end_pred_vanilla')(end_logits)
   
    embedding.trainable = False

    model = keras.Model(
        inputs=inputs,
        outputs=[start_probs, end_probs],
    )
    return model

def create_bert_custom(model_name = 'bert-base-uncased', max_len = 512, custom_layer="regular", dropout=True, inputs=[]):

  layer_options = {
    "concatenate" : layers.Concatenate(),
    "average" : layers.Average,
    "maximum" : layers.Maximum,
    "minimum" : layers.Minimum,
    "add"     : layers.Add,
    "subtract": layers.Subtract,
    "multiply": layers.Multiply,
  }

  ## BERT encoder
  encoder = TFAutoModel.from_pretrained(model_name)

  embedding = encoder(
      input_ids = inputs[0], token_type_ids=inputs[1], attention_mask=inputs[2]
  ).last_hidden_state

  start_logits = layers.Dense(1, use_bias=False, name='start_logit' + '_' + custom_layer)(embedding)
  if dropout :
    start_logits = layers.Dropout(0.5, noise_shape=None, seed=None)(start_logits)
  start_logits = layers.Flatten(name = 'flatten_start' + '_' + custom_layer)(start_logits)

  int_logits = layers.Dense(1, use_bias=False, name = 'ind_logit' + '_' + custom_layer)(embedding)
  if dropout :
    int_logits = layers.Dropout(0.5, noise_shape=None, seed=None)(int_logits)
  int_logits = layers.Flatten(name = 'flatten_int' + '_' + custom_layer)(int_logits)


  ####
  if custom_layer.lower() in layer_options:
    end_logits = layer_options[custom_layer.lower()]()([start_logits, int_logits])
  else:
    end_logits = int_logits
  ####

  start_probs = layers.Activation(keras.activations.softmax, name = 'start_pred' + '_' + custom_layer)(start_logits)
  end_probs = layers.Activation(keras.activations.softmax, name = 'end_pred' + '_' + custom_layer)(end_logits)

  model = keras.Model(
      inputs=inputs,
      outputs=[start_probs, end_probs],
  )
  return model

class EnsembleModel:
  # https://towardsdatascience.com/ensembling-convnets-using-keras-237d429157eb
  def __init__(self, models = [], inputs = []):
    outputs = [model.outputs[0] for model in models]
    self.start_probs = layers.Average()(outputs)
    outputs = [model.outputs[1] for model in models]
    self.end_probs = layers.Average()(outputs)

    self.input_len = len(models)
    self.ensemble_model = keras.Model(
        inputs = inputs,
        outputs=[self.start_probs, self.end_probs],
        name='ensemble')

  def predict(self, x_pred):
    # x_pred_ensemble = [x_final for _ in range(self.input_len) for x_final in x_pred]
    #TODO
    # relativo a print_prediction
    '''
    Now returns a List of pairs with the structure [[start, stop], [..], ..]
    '''
    pred = self.ensemble_model.predict(x_pred)

    return pred_argmax(pred)


  def plot_model(self, dpi = 55, show_shapes=True):
    return tf.keras.utils.plot_model(self.ensemble_model, dpi=dpi, show_shapes=show_shapes)

  