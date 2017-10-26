# coding: utf-8

# In[24]:

#import pip

#package_name='shutil'
#pip.main(['install', package_name])

# In[1]:

import sys
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


from keras.layers import LSTM, Dense, Embedding, Conv2D, AveragePooling2D,Flatten
from keras.models import Sequential
from optparse import OptionParser

import CNN 

argvs = sys.argv

opts, args = {}, []

print(argvs)
print("##########")

def plot_filters(layer,x,y):
    """plote the filter after the conv layer"""
    filters = layer.get_weights()[0]
    #filters = filters[:,:,:,:8]
    fig = plt.figure()
    for j in range(0,filters.shape[3]):
        ax = fig.add_subplot(y,x,j+1)
        ax.matshow(filters[:,:,0,j],cmap=matplotlib.cm.binary) # shaape [5,5,1,128]
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join('figures', 'tradSGD' + 'anma.jpg'))
    return plt

def main():

    CID = opts.cluster

    if (opts.load != 'none'): CID = opts.load

    X_train, X_test, Y_train, Y_test, X, X2, X3, enc = CNN.get_data()

    model = CNN.bulid_model(
        X_train, X_test, Y_train, Y_test, X, X2, X3, CID, fromfile='weights_8229_0_.hdf5')

    model.pop()
    model.pop()
    model.pop()

    l = model.predict(X_train)

    print(model.summary())

    print (l.shape)

    plot_filters(model.layers[0],16,8)


    return


if __name__ == "__main__":

    #plt.switch_backend('agg')
    #matplotlib.use('Agg')

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
