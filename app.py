import string
import re
import nltk
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set(style='whitegrid')

from wordcloud import WordCloud
from sklearn.metrics import classification_report,confusion_matrix

from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout
from keras.initializers import Constant

import tensorflow as tf
import warnings
warnings.simplefilter('ignore')

from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from transformers import TFBertModel,  BertConfig, BertTokenizerFast


# Name of the BERT model to use
model_name = 'cahya/bert-base-indonesian-522M'
# Max length of tokens
max_length = 150
# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
# Load the Transformers BERT model
transformer_bert_model = TFBertModel.from_pretrained(model_name, config = config)

abbertmodel = tf.keras.models.load_model('model_save')

def hapus_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

def hapus_spesial_karakter(text):
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    return text.replace("http://", " ").replace("https://", " ")

def hapus_angka(text):
    return  re.sub(r"\d+", "", str(text))

def hapus_tandabaca(text):
    return text.translate(str.maketrans(dict.fromkeys(string.punctuation, ' ')))

def hapus_whitespace_LT(text):
    return text.strip()

def hapus_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

def word_tokenize_wrapper(text):
    return word_tokenize(text)

list_stopwords = stopwords.words('indonesian')

txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)

list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

list_stopwords = set(list_stopwords)

def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

def main(original_text):
    input_text = hapus_single_char(original_text)
    input_text = hapus_spesial_karakter(input_text)
    input_text = hapus_angka(input_text)
    input_text = hapus_tandabaca(input_text)
    input_text = hapus_whitespace_LT(input_text)
    input_text = hapus_whitespace_multiple(input_text)
    tokenize_input = word_tokenize_wrapper(input_text)
    stopword_text = stopwords_removal(tokenize_input)
    sentence = ' '.join([str(item) for item in stopword_text])
    sentence = sentence.lower()

    # Set ke tokenizernya
    x_test = tokenizer(
            text=sentence,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length', 
            return_tensors='tf',
            return_token_type_ids = False,
            return_attention_mask = True,
            verbose = True)

    label_predicted_test = abbertmodel.predict(
        x={'input_ids': x_test['input_ids']},
        verbose=1
    )

    label_predicted_test['Kategori']

    mylist1 = label_predicted_test
    todf=pd.DataFrame.from_dict(mylist1['Kategori']) 
    probabilitas = todf.sort_values(by = 0, axis = 1, ascending=False)

    label_pred_max=[np.argmax(i) for i in label_predicted_test['Kategori']]

    if label_pred_max == [0]:
        print(str(label_pred_max) + " Bisnis")
    elif label_pred_max == [1]:
        print(str(label_pred_max) + " Hukum")
    elif label_pred_max == [2]:
        print(str(label_pred_max) + " Kesehatan")
    elif label_pred_max == [3]:
        print(str(label_pred_max) + " Komputer")
    elif label_pred_max == [4]:
        print(str(label_pred_max) + " Komunikasi")
    elif label_pred_max == [5]:
        print(str(label_pred_max) + " Matematika")
    elif label_pred_max == [6]:
        print(str(label_pred_max) + " Pendidikan")
    elif label_pred_max == [7]:
        print(str(label_pred_max) + " Pertanian")
    elif label_pred_max == [8]:
        print(str(label_pred_max) + " Teknik")
    else:
        print('Not Classify')
    
if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='testingprediktion')
    parser.add_argument('abstrak')
    args = parser.parse_args()
    main(args.abstrak)
