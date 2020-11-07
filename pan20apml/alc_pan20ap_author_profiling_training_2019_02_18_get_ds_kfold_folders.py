#!/usr/bin/env python
# coding: utf-8

import os
import shlex
import subprocess
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

xmls_base_directory = '../../pan20ap-ds/pan20-author-profiling-training-2020-02-23'

n_splits = 10


for lang in ['es', 'en']:
    truth_path = '{}/{}/truth.txt'.format(xmls_base_directory,lang)
    print(truth_path)
    df = pd.read_csv(truth_path, sep=':::', header=None, names=['author_id', 'label'])
    df.head()

    duplicateRowsDF = df[df.duplicated(['author_id'])]
    # duplicateRowsDF

    df['label'].value_counts()

    X = df['author_id']
    y = df['label']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    skf.get_n_splits(X, y)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        truth_path_train = '{}/{}/truth-k{}-{}-train.txt'.format(xmls_base_directory, lang, n_splits, i)
        print(i, truth_path_train)
        df_train = df.iloc[train_index]
        df_train.to_csv(truth_path_train, sep=':', header=None, index=False, columns=['author_id', None, None,'label'])
        path_train = "{}-{}-{}-{}/{}".format(
                xmls_base_directory,
                'k{}'.format(n_splits),
                i,
                'train',
                lang,
            )
        cmd = "mkdir -p {} && cat {} | awk -F: '{{print $1}}' | xargs -n1 -I{{}} cp {}/{{}}.xml {}".format(
            path_train,
            truth_path_train,
            '{}/{}'.format(xmls_base_directory, lang),
            path_train,
        )
        print(cmd)
        stream = os.popen(cmd)
        print(stream.read())

        cmd = "cp {} {}".format(
            truth_path_train,
            '{}/truth.txt'.format(path_train),
        )
        print(cmd)
        stream = os.popen(cmd)
        print(stream.read())
        
        truth_path_dev = '{}/{}/truth-k{}-{}-dev.txt'.format(xmls_base_directory, lang, n_splits, i)
        print(i, truth_path_dev)
        df_dev = df.iloc[test_index]
        df_dev.to_csv(truth_path_dev, sep=':', header=None, index=False, columns=['author_id', None, None,'label'])
        path_dev = "{}-{}-{}-{}/{}".format(
                xmls_base_directory,
                'k{}'.format(n_splits),
                i,
                'dev',
                lang,
            )
        cmd = "mkdir -p {} && cat {} | awk -F: '{{print $1}}' | xargs -n1 -I{{}} cp {}/{{}}.xml {}".format(
            path_dev,
            truth_path_dev,
            '{}/{}'.format(xmls_base_directory, lang),
            path_dev,
        )
        print(cmd)
        stream = os.popen(cmd)
        print(stream.read())

        cmd = "cp {} {}".format(
            truth_path_dev,
            '{}/truth.txt'.format(path_dev),
        )
        print(cmd)
        stream = os.popen(cmd)
        print(stream.read())
