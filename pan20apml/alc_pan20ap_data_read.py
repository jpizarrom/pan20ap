#!/usr/bin/env python
# coding: utf-8

import glob
import os
import pandas as pd

from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

from xml.etree import ElementTree

#def extract_features(docs_train, params):
#    
#    vectorizer = _objective_feats(params)
#
#    vectorizer.fit(docs_train)
#    return vectorizer

from pan20ap_utils import load_df_train, load_df_dev, read_xmls_v0, read_xmls_v1, read_xmls_v2

import glob
import os
import re
#import neptune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from emoji import demojize
from xml.etree import ElementTree
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tensorflow.keras.callbacks import History
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.keras.utils import plot_model
#from keras_mlflow import NeptuneMonitor
import wandb
from wandb.keras import WandbCallback

read_xmlss = {
    'read_xmls_v0': read_xmls_v0,
    'read_xmls_v1': read_xmls_v1,
    'read_xmls_v2': read_xmls_v2,
}

save_df = True
langs = [
    'en',
    'es',
]

ds_name_folds = []

#ds_name_folds += ['-k5-0', '-k5-1', '-k5-2', '-k5-3', '-k5-4']
ds_name_folds += ['-k10-{}'.format(i) for i in range(10)]
# ds_name_folds += ['']
print(ds_name_folds)

for lang in langs:
    for read_xmls in read_xmlss:
        print('='*80, read_xmls)
        for ds_name_fold in ds_name_folds:
            model_suffix = '.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a{}.{}'.format(ds_name_fold, read_xmls)
            print('='*40, 'train')
            xmls_directory_train = '../../pan20ap-ds/pan20-author-profiling-training-2020-02-23{}-train/{}'.format(ds_name_fold, lang)
            truth_path_train = '../../pan20ap-ds/pan20-author-profiling-training-2020-02-23{}-train/{}/truth.txt'.format(ds_name_fold, lang)
            print(xmls_directory_train)
            print(xmls_directory_train)
            df_train = load_df_train(xmls_directory_train, lang, model_suffix, truth_path_train, read_xmls=read_xmlss[read_xmls], save_df=save_df)
            print(df_train.shape)
            print(df_train.head())

            print('='*40, 'dev')
            xmls_directory_dev = '../../pan20ap-ds/pan20-author-profiling-training-2020-02-23{}-dev/{}'.format(ds_name_fold, lang)
            truth_path_dev = '../../pan20ap-ds/pan20-author-profiling-training-2020-02-23{}-dev/{}/truth.txt'.format(ds_name_fold, lang)
            print(xmls_directory_dev)
            print(truth_path_dev)
            df_dev = load_df_dev(xmls_directory_dev, lang, model_suffix, truth_path_dev, read_xmls=read_xmlss[read_xmls], save_df=save_df)
            print(df_dev.shape)
            print(df_dev.head())
