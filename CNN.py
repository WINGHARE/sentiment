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

argvs = sys.argv

opts, args = {}, []

print(argvs)
print("##########")


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


def f1score(y_true, y_pred):
    r = recall(y_true, y_pred)
    p = precision(y_true, y_pred)
    return 2 * p * r / (p + r)


def decode_y(y, features=np.array([0, 1])):
    return np.dot(y, features).astype(int)


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

    Y = [sentiment[i] for i in range(0, len(sentiment))]
    Y = np.asarray(Y)

    ohenc = OneHotEncoder()
    Y2 = ohenc.fit_transform(Y.reshape(-1, 1)).toarray()

    X_train, X_test, Y_train, Y_test = train_test_split(X3, Y2, test_size=0.4)
    return X_train, X_test, Y_train, Y_test, X, X2, X3, ohenc


class TestCallback(Callback):
    def __init__(self, test_data1, test_data2):
        self.Xdata = test_data1
        self.Ydata = test_data2

    def on_epoch_end(self, epoch, logs={}):
        x = self.Xdata
        y = self.Ydata
        loss, acc, recall, pre, f1 = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def bulid_model(X_train,
                X_test,
                Y_train,
                Y_test,
                X,
                X2,
                X3,
                CID,
                fromfile='none'):
    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation='relu',
            input_shape=(X3[0].shape[0],X3[0].shape[1],1)))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(2, activation='softmax'))

    if (fromfile == 'none'):
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', recall, precision, f1score])
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

        model.save_weights(
            filepath=os.path.join('tmp', 'weights_' + CID + '.hdf5'))
        return model

    else:
        filepath = os.path.join('tmp', fromfile)
        model.load_weights(filepath=filepath)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', recall, precision, f1score])
        print(model.summary())
        return model

    return model


def main():

    CID = opts.cluster

    if (opts.load != 'none'): CID = opts.load

    X_train, X_test, Y_train, Y_test, X, X2, X3, enc = get_data()

    model = bulid_model(
        X_train, X_test, Y_train, Y_test, X, X2, X3, CID, fromfile=opts.load)

    newData = X_test.reshape(X_test.shape[0], 1, 100, 20)

    Y_score = model.predict_proba(X_test)

    roc.roc_plot(
        Y_test, Y_score, 2, filepath=os.path.join('figures', CID + 'roc.jpg'))

    Y_de = decode_y(Y_test, features=enc.active_features_)
    Y_pred = model.predict(X_test)
    Y_depred = decode_y(Y_pred, features=enc.active_features_)
    print(classification_report(Y_de, Y_depred))

    return


if __name__ == "__main__":

    plt.switch_backend('agg')
    mp.use('Agg')

    op = OptionParser()
    op.add_option(
        '-c',
        '--cluster',
        action='store',
        type='string',
        dest='cluster',
        help='indicate the clusterid')
    op.add_option(
        '-d',
        '--date',
        action='store',
        type='string',
        dest='date',
        help='indicate the date')
    op.add_option(
        '-l',
        '--load',
        default='none',
        action='store',
        type='string',
        dest='load',
        help='load weight form file')
    (opts, args) = op.parse_args()
    if len(args) > 0:
        op.print_help()
        op.error('Please input options instead of arguments.')
        exit(1)

    main()
