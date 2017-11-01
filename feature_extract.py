import os
import re
import sys
import time

import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import roc as roc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import LSTM, Dense, Embedding, Conv2D, AveragePooling2D,Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from optparse import OptionParser
from sklearn.metrics import classification_report

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

words = set(nltk.corpus.words.words())

def removeNonEnglish(s):
    sen = s
    print(" ".join(w for w in nltk.wordpunct_tokenize(sen) if w.lower() in words or not w.isalpha()))
    return " ".join(w for w in nltk.wordpunct_tokenize(sen) if w.lower() in words or not w.isalpha())

def removeStopWords(ss):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(ss)
    sent = [w for w in tokens if not w in stop_words]
    sent2 = []
    for w in tokens:
        if w not in stop_words:
            sent2.append(w)
    return " ".join(sent2)

def lemmat(ss):
    lem = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(ss)
    sent = [lem.lemmatize(tokens[i],'n')for i in range(len(tokens))]
    sent2 = [lem.lemmatize(sent[i],'v')for i in range(len(sent))]
    return " ".join(sent2)

def get_data():
    data = pd.read_csv(os.path.join('data', '01.csv'), encoding="ISO-8859-1")
    text = data['text']
    sentiment = data['target']
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply(
        (lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    text_list = [str(s.encode('ascii')) for s in text.values]
    tokenizer.fit_on_texts(text_list)
    X = tokenizer.texts_to_sequences(text_list)
    X = pad_sequences(X)
    X2 = tokenizer.texts_to_matrix(text_list, mode="tfidf")
    X3 = [np.reshape(X2[i], (-1, 20)) for i in range(0, len(X2))]
    X3 = np.asarray(X3)
    X3 = X3.reshape(X3.shape[0],X3.shape[1],X3.shape[2],1)

    Y = [sentiment[i] for i in range(0, len(sentiment))]
    Y = np.asarray(Y)

    ohenc = OneHotEncoder()
    Y2 = ohenc.fit_transform(Y.reshape(-1, 1)).toarray()

    X_train, X_test, Y_train, Y_test = train_test_split(X3, Y2, test_size=0.4)
    return X_train, X_test, Y_train, Y_test, X, X2, X3, ohenc


def get_data2():
    data = pd.read_csv(os.path.join('data', '01.csv'), encoding="ISO-8859-1")
    text = data['text']
    sentiment = data['target']
    data.head()
    text = text.apply(lambda x: x.lower()) #lowercase
    text = text.apply((lambda x: re.sub(r'[?|$|&|*|%|@|(|)|~]','',x))) #remove punctuations
    text = text.apply((lambda x: re.sub(r'[^a-zA-Z0-9 ]','',x))) #only numbers and alphabet
    text = text.apply(lambda x:removeStopWords(x)) #remove stopwords
    text = text.apply(lambda x:lemmat(x)) #lemmatize the sentence

    text = text.apply(lambda x: removeNonEnglish(x)) #remove

    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    text_list = [str(s.encode('ascii')) for s in text.values]
    tokenizer.fit_on_texts(text_list)
    X = tokenizer.texts_to_sequences(text_list)
    X = pad_sequences(X)
    X2 = tokenizer.texts_to_matrix(text_list, mode="tfidf")
    X3 = [np.reshape(X2[i], (-1, 20)) for i in range(0, len(X2))]
    X3 = np.asarray(X3)
    X3 = X3.reshape(X3.shape[0],X3.shape[1],X3.shape[2],1)

    Y = [sentiment[i] for i in range(0, len(sentiment))]
    Y = np.asarray(Y)

    ohenc = OneHotEncoder()
    Y2 = ohenc.fit_transform(Y.reshape(-1, 1)).toarray()

    X_train, X_test, Y_train, Y_test = train_test_split(X3, Y2, test_size=0.4)
    return X_train, X_test, Y_train, Y_test, X, X2, X3, ohenc