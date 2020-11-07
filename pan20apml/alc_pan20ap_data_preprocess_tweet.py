#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from xml.etree import ElementTree
import os

from sklearn.metrics import classification_report

save_df = True
langs = [
    'en',
    'es',
]
model_suffix = '.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a'
#ds_name_folds = ['-k5-0.read_xmls_v0', '-k5-1.read_xmls_v0', '-k5-2.read_xmls_v0', '-k5-3.read_xmls_v0', '-k5-4.read_xmls_v0', '.read_xmls_v0']
ds_name_folds = ['-k10-{}.read_xmls_v0'.format(i) for i in range(10)]
#ds_name_folds = ['.read_xmls_v0']
print(langs, model_suffix, ds_name_folds)

from pan20ap_utils import preprocess_tweet_funcs

preprocess_tweet_funcs_k = [
            'preprocess_tweet_v0_0',
            'preprocess_tweet_v0_1',
            'preprocess_tweet_v1_0',
            'preprocess_tweet_v1_1',
            'preprocess_tweet_v2_0',
            'preprocess_tweet_v2_0_1',
            'preprocess_tweet_v2_1',
            'preprocess_tweet_v2_1_1',
            'preprocess_tweet_v3_0',
            'preprocess_tweet_v3_0_1',
            'preprocess_tweet_v3_1',
            'preprocess_tweet_v3_1_1',
]

#df_train.en.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-0.pkl
def read_df(df_name,lang, model_suffix, ds_name_fold):
    fname = './out/{}.{}{}{}.pkl'.format(
        df_name,
        lang,
        model_suffix,
        ds_name_fold,
    )
    print(fname)
    df = pd.read_pickle(fname)
    return df

def apply_preprocess_tweet_funcs(df, df_name, lang, model_suffix, ds_name_fold, preprocess_tweet_func, save_df=False):
    fname = './out/{}.{}{}{}.{}.pkl'.format(df_name, lang, model_suffix, ds_name_fold, preprocess_tweet_func)
    print(fname)
    try:
        raise FileNotFoundError
    #    df_train = pd.read_pickle(fname)
    except FileNotFoundError:
        df['tweet'] = df['tweet'].apply(preprocess_tweet_funcs[preprocess_tweet_func])
        if save_df:
            if os.path.exists(fname):
                print('save_df exists', fname)
            else:
                print('save_df not exists', fname)
                df.to_pickle(fname)

    print(df.shape)
    return df

for lang in langs:
    for preprocess_tweet_func in preprocess_tweet_funcs_k:
        print('='*80, preprocess_tweet_func)
        for ds_name_fold in ds_name_folds:
            for df_name in ['df_train', 'df_dev']:
                print('='*10,preprocess_tweet_func, ds_name_fold, df_name)
                df = read_df(df_name=df_name, lang=lang, model_suffix=model_suffix, ds_name_fold=ds_name_fold)
                print(df.head(2))

                df = apply_preprocess_tweet_funcs(
                    df=df, df_name=df_name,
                    lang=lang, model_suffix=model_suffix, ds_name_fold=ds_name_fold, preprocess_tweet_func=preprocess_tweet_func, save_df=save_df)
                print(df.head(2))

