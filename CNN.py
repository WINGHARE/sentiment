# coding: utf-8

# In[24]:

#import pip

#package_name='shutil'
#pip.main(['install', package_name])

# In[1]:

import os
import re
import sys
import time

import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import LSTM, Dense, Embedding, GRU
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

argvs = sys.argv

CID = sys.argv[1]

def precision(y_true, y_pred):
    #Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision1(y_true, y_pred):
    return precision_score(y_true, y_pred)


def recall1(y_true, y_pred):
    return recall_score(y_true, y_pred)

def get_data():
    data = pd.read_csv(os.path.join('data', '01.csv'), encoding="ISO-8859-1")
    text = data['text']
    sentiment = data['target']
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply(
        (lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    text_list = [s.encode('ascii') for s in text.values]
    tokenizer.fit_on_texts([s.encode('ascii') for s in text_list])
    X = tokenizer.texts_to_sequences(text_list)
    X = pad_sequences(X)
    X2 = tokenizer.texts_to_matrix(text_list, mode="tfidf")
    X3 = [np.reshape(X2[i], (-1, 20)) for i in range(0, len(X2))]
    X3 = np.asarray(X3)

    Y = [sentiment[i] for i in range(0, len(sentiment))]
    Y = np.asarray(Y)

    ohenc= OneHotEncoder()
    Y2 = ohenc.fit_transform(Y.reshape(-1,1)).toarray()

    X_train,X_test,Y_train,Y_test = train_test_split(X3,Y2,test_size=0.25)
    return X_train,X_test,Y_train,Y_test,X,X2,X3

class TestCallback(Callback):
    def __init__(self, test_data1, test_data2):
        self.Xdata = test_data1
        self.Ydata = test_data2

    def on_epoch_end(self, epoch, logs={}):
        x = self.Xdata
        y = self.Ydata
        loss, acc, recall, pre = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

def main():

    X_train,X_test,Y_train,Y_test,X,X2,X3 = get_data()
    model = Sequential()
    model.add(
        GRU(512,
        return_sequences=True,
        input_shape=X3[0].shape,
        dropout=0.2,
        recurrent_dropout=0.2))   
    model.add(GRU(512, return_sequences=False, dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        #metrics=['accuracy', recall, precision])
        metrics=['accuracy', 'recall', 'precision','f1score'])
    print(model.summary())

    batch_size = 32

    checkpointer = ModelCheckpoint(
        filepath=os.path.join('tmp', 'weights_' + CID + '.hdf5'),
        verbose=1,
        save_best_only=True)
    
    model.fit(
        X_train,
        Y_train,
        epochs=10,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_test, Y_test),
        callbacks=[checkpointer])
    
    newData = X_test.reshape(183, 1, 100, 20)

    model = Sequential()
    model.add(
        GRU(512,
        return_sequences=True,
        input_shape=X3[0].shape,
        dropout=0.2,
        recurrent_dropout=0.2))
    
    model.add(GRU(512, return_sequences=False, dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.load_weights(filepath=os.path.join('tmp', 'weights_' + CID + '.hdf5'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy', 'recall', 'precision','f1score'])
    print(model.summary())
    model.save_weights(filepath=os.path.join('tmp', 'weights_' + CID + '.hdf5'))


    pcount, ncount, ppredict, npredict = 0, 0, 0, 0
    for x in range(len(newData)):
        result = model.predict(newData[x])[0]
        if np.argmax(result) == np.argmax(Y_test[x]):
            if np.argmax(Y_test[x]) == 0:
                ppredict += 1
            else:
                npredict += 1

        if np.argmax(Y_test[x]) == 0:
            pcount += 1
        else:
            ncount += 1

    print("True 1:\t", ppredict / pcount)
    print("False 1:\t", 1 - ppredict / pcount)
    print("True 0:\t", npredict / ncount)
    print("False 0:\t", 1 - npredict / ncount)

    plt.switch_backend('agg')
    mp.use('Agg')

    cm = [[npredict / ncount, 1 - npredict / ncount],
    [ppredict / pcount, 1 - ppredict / pcount]]
    labels = ['0', '1']

    fig, ax = plt.subplots()
    h = ax.matshow(cm)
    fig.colorbar(h)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    ax.set_xlabel('Ground truth')
    ax.set_ylabel('Predicted')
    plt.savefig(os.path.join('figures', str(argvs[1]) + '.jpg'))
    plt.show()
    plt.close()
    return


if __name__ == "__main__":
    main()

