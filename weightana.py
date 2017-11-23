# coding: utf-8

# In[24]:

#import pip

#package_name='shutil'
#pip.main(['install', package_name])

# In[1]:
# cd D:/pyws/sentiment

import sys
import os
import re

import pydot
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


from keras.layers import LSTM, Dense, Embedding, Conv2D, AveragePooling2D,Flatten
from keras.models import Sequential
from optparse import OptionParser
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model


import feature_extract as f 
import CNN
import CNN_MAX

argvs = sys.argv

opts, args = {}, []

os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'


print(argvs)
print("##########")

def plot_filters(layer,x,y,filepath='filters.jpg'):
    """plote the filter after the conv layer"""
    filters = layer.get_weights()[0]
    #filters = filters[:,:,:,:8]
    fig = plt.figure()
    for j in range(0,filters.shape[3]):
        ax = fig.add_subplot(y,x,j+1)
        ax.matshow(filters[:,:,0,j],cmap=mp.cm.binary) # shaape [5,5,1,128]
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.savefig(filepath)
    plt.tight_layout()
    plt.show()
    plt.close()
    return plt

def plot_conv(output,x,y,filepath='cov.jpg'):
    """plote the filter after the conv layer"""
    filters = output
    #filters = filters[:,:,:,:8]
    fig = plt.figure(figsize=(18,9),dpi=100)
    for j in range(0,filters.shape[3]):
        ax = fig.add_subplot(y,x,j+1)
        ax.imshow(filters[0,:,:,j]) # shaape [5,5,1,128]
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    #plt.tight_layout()
    plt.savefig(filepath)
    plt.show()
    plt.close()
    return plt

def get_dict():
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
    return tokenizer


def main():

    CID = opts.cluster

    if (opts.load != 'none'): CID = opts.load

    X_train, X_test, Y_train, Y_test, X, X2, X3, enc = f.get_data2()

    model = CNN.bulid_model(
        X_train, X_test, Y_train, Y_test, X, X2, X3, CID, fromfile='weights_8401_0_.hdf5')

    plot_model(model,to_file=os.path.join('figures', 'model_' + '8401.png'))

    model.pop()
    model.pop()
    model.pop()

    l = model.predict(X_train)

    print(model.summary())

    print (l.shape)

    plot_filters(model.layers[0],16,8,filepath=os.path.join('figures', 'filters' + '8401.jpg'))

    tk = get_dict()
    
    allwords = ' '.join(list(tk.word_index.keys()))
    
    vec=tk.texts_to_matrix([allwords], mode="tfidf")

    vec = vec.reshape(1,100,20,1)

    poolresult = model.predict(vec)

    plt.imshow(vec[0,:,:,0])
    plt.show()

    plot_conv(poolresult,16,8,filepath=os.path.join('figures', 'covs_pooled_max' + '8401.jpg'))

    model.pop()
    result = model.predict(vec)

    plot_conv(result,16,8,filepath=os.path.join('figures', 'covs_max' + '8401.jpg'))


    return


if __name__ == "__main__":

    #plt.switch_backend('agg')
    #mp.use('Agg')

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
