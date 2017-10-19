import os
import re
import sys
import time

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer,
                                             TfidfTransformer)
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import (ShuffleSplit, learning_curve,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.preprocessing import OneHotEncoder


import roc as roc


data=pd.read_csv(os.path.join('data','01.csv'),encoding = "ISO-8859-1");


data['text'] = data['text'].map(lambda x: x.lower());
target = data.as_matrix(['target']);
text = data.as_matrix(['text']);

train_data,test_data,train_target,test_target = train_test_split( 
text.ravel(), target.ravel(), test_size=0.40, random_state=42)


"""
try doing lemmatizer

"""
class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



#######
# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
######## 



clf2 = Pipeline([('vect', CountVectorizer(tokenizer=tokenize,ngram_range=(1, 2))),
                       ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='log',random_state=42,shuffle=True, alpha=0.0001*0.75, penalty='l1',
                                            n_iter=20 )),
 ])

_ = clf2.fit(train_data, train_target)
predicted = clf2.predict(test_data)
proba = clf2.predict_proba(test_data)
ohenc= OneHotEncoder()
Y2 = ohenc.fit_transform(test_target.reshape(-1,1)).toarray()
print(proba)
roc.roc_plot(Y2,proba,2,filepath=os.path.join('figures', 'tradSGD' + 'roc.jpg'))

print(metrics.classification_report(test_target, predicted,))


"""

try SVC

"""

clf5 = Pipeline([('vect', CountVectorizer(tokenizer=tokenize,ngram_range=(1, 2))),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SVC(C=1.25, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=1, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)),
 ])

_ = clf5.fit(train_data, train_target);
predicted = clf5.predict(test_data)
#proba= clf5.predict_proba(test_data)
print(metrics.classification_report(test_target, predicted,))
