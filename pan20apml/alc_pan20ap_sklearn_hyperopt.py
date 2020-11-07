#!/usr/bin/env python
# coding: utf-8

import argparse
import hashlib
import importlib
import json
import os
from distutils.util import strtobool

# if wandb_enable:
import wandb

# wandb._get_python_type()# != "python"

from hyperopt import STATUS_OK

import glob
import os
import pandas as pd

from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

from pan20ap_utils import (
    load_df_train,
    load_df_dev,
    preprocess_tweet_funcs,
    read_xmls_funcs,
)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.base import BaseEstimator, TransformerMixin
import re
import emoji
import string

import pickle
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

def conf_wandb(space):
    params = {}
    params["run_suffix"] = space["fmin"]["params"]["run_suffix"]
    #    wandb.config.['wandb_enable'] = wandb_enable
    params["wandb_save_model_enable"] = space["fmin"]["params"][
        "wandb_save_model_enable"
    ]
    params["lang"] = space["fmin"]["params"]["lang"]
    params["task"] = space["fmin"]["params"]["task"]
    #    params['train_name'] = space['fmin']['params']['xmls_base_directory_train_name']
    params["ds_name"] = space["fmin"]["params"]["ds_name"]
    params["ds_name_folds"] = space["fmin"]["params"]["ds_name_folds"]
    #    params['max_evals'] = space['fmin']['params']['max_evals']
    params["hash_object"] = space["fmin"]["params"]["hash_object"]
    params["model_suffix"] = space["fmin"]["params"]["model_suffix"]

    return params

class TextCounts(BaseEstimator, TransformerMixin):
    # https://towardsdatascience.com/sentiment-analysis-with-text-mining-13dd2b33de27

    def count_regex(self, pattern, tweet):
        return len(re.findall(pattern, tweet))

    def fit(self, X, y=None, **fit_params):
        # fit method is used when specific operations need to be done on the train data, but not on the test data
        return self

    def transform(self, X, **transform_params):
        count_words = X.apply(lambda x: self.count_regex(r"\w+", x))
        #        count_mentions = X.apply(lambda x: self.count_regex(r'@\w+', x))
        count_mentions = X.apply(lambda x: self.count_regex(r"#USER#", x))
        # count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+', x))
        count_hashtags = X.apply(lambda x: self.count_regex(r"#HASHTAG#", x))
        count_capital_words = X.apply(lambda x: self.count_regex(r"\b[A-Z]{2,}\b", x))
        count_excl_quest_marks = X.apply(lambda x: self.count_regex(r"!|\?", x))
        count_urls = X.apply(lambda x: self.count_regex(r"#URL#", x))
        #        count_urls = X.apply(lambda x: self.count_regex(r'http.?://[^\s]+[\s]?', x))
        # We will replace the emoji symbols with a description, which makes using a regex for counting easier
        # Moreover, it will result in having more words in the tweet
        count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(
            lambda x: self.count_regex(r":[a-z_&]+:", x)
        )
        #        count_emojis = X.apply(lambda x: self.count_regex(r'xxemj', x))
        count_punctuation = X.apply(
            lambda x: len([c for c in str(x) if c in string.punctuation])
        )

        df = pd.DataFrame(
            {
                "count_words": count_words,
                "count_mentions": count_mentions,
                "count_hashtags": count_hashtags,
                "count_capital_words": count_capital_words,
                "count_excl_quest_marks": count_excl_quest_marks,
                "count_urls": count_urls,
                "count_emojis": count_emojis,
                "count_punctuation": count_punctuation,
            }
        )

        return df


def get_model(space):
    if space["classifier"]["name"] == "LinearSVC":
        model = LinearSVC(**space["classifier"]["params"])
    #        model = CalibratedClassifierCV(model)
    #        model = SklearnClassifier(SVC(**space['classifier']['params'], kernel='linear', probability=True))

    elif space["classifier"]["name"] == "LogisticRegression":
        model = LogisticRegression(**space["classifier"]["params"])

    elif space["classifier"]["name"] == "MultinomialNB":
        model = MultinomialNB(**space["classifier"]["params"])

    elif space["classifier"]["name"] == "RandomForestClassifier":
        model = RandomForestClassifier(**space["classifier"]["params"])

    elif space["classifier"]["name"] == "XGBClassifier":
        import xgboost

        model = xgboost.XGBClassifier(**space["classifier"]["params"])

    else:
        raise NotImplemented
    return model


def _objective_clf(space, X_train, y_train, X_dev, y_dev):

    model = get_model(space)
    print(model)

    if space["classifier"]["name"] == "XGBClassifier":
        eval_set = [(X_train, y_train), (X_dev, y_dev)]

        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric="error",
            early_stopping_rounds=30,
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_dev)

    #    wandb.sklearn.plot_learning_curve(model, X_dev, y_dev)
    #    wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_dev, y_dev)
    #    wandb.sklearn.summary_metrics(model, X_train, y_train, X_dev, y_dev)

    accuracy = metrics.accuracy_score(y_dev, y_pred)
    (
        precisions_weighted,
        recalls_weighted,
        f_measures_weighted,
        support_weighted,
    ) = metrics.precision_recall_fscore_support(y_dev, y_pred, average="weighted")
    # precisions_macro, recalls_macro, f_measures_macro, support_macro = metrics.precision_recall_fscore_support(y_dev, y_pred, average='macro')
    # precisions_micro, recalls_micro, f_measures_micro, support_micro  = metrics.precision_recall_fscore_support(y_dev, y_pred, average='micro')

    results = {
        "loss": -accuracy,
        "status": STATUS_OK,
        "metric_classifier_accuracy": accuracy,
        "metric_classifier_precision_weighted": precisions_weighted,
        "metric_classifier_recall_weighted": recalls_weighted,
        "metric_classifier_fscore_weighted": f_measures_weighted,
        "metric_classifier_support_weighted": support_weighted,
        #    'metric_classifier_precision_macro':  precisions_macro,
        #    'metric_classifier_recall_macro': recalls_macro,
        #    'metric_classifier_fscore_macro': f_measures_macro,
        #    'metric_classifier_support_macro': support_macro,
        #    'metric_classifier_precision_micro':  precisions_micro,
        #    'metric_classifier_recall_micro': recalls_micro,
        #    'metric_classifier_fscore_micro': f_measures_micro,
        #    'metric_classifier_support_micro': support_micro,
        "attachments": {"clf": pickle.dumps(model)},
    }
    return results


def _objective_feats(space):
    features = []

    if "word" in space["feats"]["name"]:
        word_vectorizer = TfidfVectorizer(
            preprocessor=preprocess_tweet_funcs[
                space["feats"]["params"]["word"]["preprocessor"]
            ],
            tokenizer=TweetTokenizer().tokenize,
            analyzer="word",
            stop_words=space["feats"]["params"]["word"]["stop_words"],
            ngram_range=space["feats"]["params"]["word"]["ngram_range"],
            min_df=space["feats"]["params"]["word"]["min_df"],
            max_df=space["feats"]["params"]["word"]["max_df"],
            use_idf=True,
            sublinear_tf=True,
        )

        if "svd" in space["feats"]["name"] and False:
            features.append(
                (
                    "word_ngram",
                    Pipeline(
                        [
                            ("tfidf", word_vectorizer),
                            ("svd", TruncatedSVD(**space["feats"]["params"]["svd"])),
                        ]
                    ),
                )
            )
        else:
            features.append(("word_ngram", word_vectorizer))

    if "char" in space["feats"]["name"]:
        char_vectorizer = TfidfVectorizer(
            preprocessor=preprocess_tweet_funcs[
                space["feats"]["params"]["char"]["preprocessor"]
            ],
            analyzer="char",
            ngram_range=space["feats"]["params"]["char"]["ngram_range"],
            min_df=space["feats"]["params"]["char"]["min_df"],
            max_df=space["feats"]["params"]["char"]["max_df"],
            use_idf=True,
            sublinear_tf=True,
        )

        if "svd" in space["feats"]["name"] and False:
            features.append(
                (
                    "char_ngram",
                    Pipeline(
                        [
                            ("tfidf", char_vectorizer),
                            ("svd", TruncatedSVD(**space["feats"]["params"]["svd"])),
                        ]
                    ),
                )
            )
        else:
            features.append(("char_ngram", char_vectorizer))

    if "eda" in space["feats"]["name"]:
        tc = TextCounts()
        features.append(("textcounts", tc))

    if "pos" in space["feats"]["name"]:
        pos_vec = CountVectorizer(
            vocabulary=df_pos, ngram_range=(1, 3), token_pattern=r"(?u)\b\w+\b"
        )
        features.append(("pos_vec", pos_vec))

    if "neg" in space["feats"]["name"]:
        neg_vec = CountVectorizer(
            vocabulary=df_neg, ngram_range=(1, 3), token_pattern=r"(?u)\b\w+\b"
        )
        features.append(("neg_vec", neg_vec))

    print(features)
    if "svd" in space["feats"]["name"]:
        vectorizer = Pipeline(
            [
                ("feats", FeatureUnion(features)),
                ("svd", TruncatedSVD(**space["feats"]["params"]["svd"])),
            ]
        )
    else:
        vectorizer = Pipeline([("feats", FeatureUnion(features))])
    return vectorizer

# import xgboost


# def objective_clf(space):
#     print(space)
#     if task == 'label':
#         return _objective_clf(space, X_train, y_train, X_dev, y_dev)
#     else:
#         raise


def objective_feats_and_clf(space):
    print("=" * 40, "objective_feats_and_clf")
    params = {}  #  conf_wandb(space)

    for k in space:
        params["{}-name".format(k)] = space[k]["name"]
        for l in space[k]["params"]:
            if type(space[k]["params"][l]) != dict:
                params["{}-{}".format(k, l)] = space[k]["params"][l]
            else:
                for m in space[k]["params"][l]:
                    params["{}-{}-{}".format(k, l, m)] = space[k]["params"][l][m]

    if space["fmin"]["params"]["wandb_enable"]:
        wandb.init(
            project="pan20apml", config=params, reinit=True, allow_val_change=True
        )

    print("space", space)

    ds_name = space["fmin"]["params"]["ds_name"]
    ds_name_folds = space["fmin"]["params"]["ds_name_folds"]
    rets = []
    for ds_name_fold in ds_name_folds:
        # train df
        xmls_base_directory_train_name = "{}{}-train".format(ds_name, ds_name_fold)
        xmls_base_directory_train = "{}/{}".format(
            space["fmin"]["params"]["xmls_base_directory"],
            xmls_base_directory_train_name,
        )
        # xmls_base_directory_train = space['fmin']['params']['xmls_base_directory_train']
        lang = space["fmin"]["params"]["lang"]
        model_suffix = "{}{}".format(
            space["fmin"]["params"]["model_suffix"], ds_name_fold
        )
        print("lang", lang)
        print("model_suffix", model_suffix)

        xmls_directory_train = "{}/{}".format(xmls_base_directory_train, lang)
        truth_path_train = "{}/truth.txt".format(xmls_directory_train)
        print("xmls_directory_train", xmls_directory_train)
        print("truth_path_train", truth_path_train)

        read_xmls = read_xmls_funcs[
            space["fmin"]["params"]["read_xmls"].replace(".", "")
        ]
        print("read_xmls", space["fmin"]["params"]["read_xmls"].replace(".", ""))
        df_train = load_df_train(
            xmls_directory_train,
            lang,
            model_suffix,
            truth_path_train,
            read_xmls=read_xmls,
        )
        docs_train_clean = df_train["tweet"]
        y_train = df_train[space["fmin"]["params"]["task"]]

        # dev df
        xmls_base_directory_dev_name = "{}{}-dev".format(ds_name, ds_name_fold)
        xmls_base_directory_dev = "{}/{}".format(
            space["fmin"]["params"]["xmls_base_directory"], xmls_base_directory_dev_name
        )
        # xmls_base_directory_dev = space['fmin']['params']['xmls_base_directory_dev']
        xmls_directory_dev = "{}/{}".format(xmls_base_directory_dev, lang)
        truth_path_dev = "{}/truth.txt".format(xmls_directory_dev)
        print("xmls_directory_dev", xmls_directory_dev)
        print("truth_path_dev", truth_path_dev)
        #    print('df_dev.{}{}.pkl'.format(lang, model_suffix))

        df_dev = load_df_dev(
            xmls_directory_dev, lang, model_suffix, truth_path_dev, read_xmls=read_xmls,
        )
        docs_dev_clean = df_dev["tweet"]

        y_dev = df_dev[space["fmin"]["params"]["task"]]

        vectorizer = _objective_feats(space)

        X_train = vectorizer.fit_transform(docs_train_clean)
        X_dev = vectorizer.transform(docs_dev_clean)

        if space["fmin"]["params"]["task"] == "label":
            ret = _objective_clf(space, X_train, y_train, X_dev, y_dev)
            for k in ret.keys():
                if k not in ["attachments"]:
                    print(k, ret[k])
            ret["attachments"]["vect"] = pickle.dumps(vectorizer)
            rets.append(ret)
        else:
            raise
    for k in rets[0].keys():
        if k not in ["attachments"]:
            print(k, [x[k] for x in rets])

    results = {
        "loss": np.mean([x["loss"] for x in rets]),
        "status": STATUS_OK,
        "metric_classifier_accuracy": np.mean(
            [x["metric_classifier_accuracy"] for x in rets]
        ),
        "metric_classifier_accuracy_std": np.std(
            [x["metric_classifier_accuracy"] for x in rets]
        ),
        "metric_classifier_precision_weighted": np.mean(
            [x["metric_classifier_precision_weighted"] for x in rets]
        ),
        "metric_classifier_recall_weighted": np.mean(
            [x["metric_classifier_recall_weighted"] for x in rets]
        ),
        "metric_classifier_fscore_weighted": np.mean(
            [x["metric_classifier_fscore_weighted"] for x in rets]
        ),
        "metric_classifier_fscore_weighted_std": np.std(
            [x["metric_classifier_fscore_weighted"] for x in rets]
        ),
        #'metric_classifier_support_weighted': np.mean([x['metric_classifier_support_weighted'] for x in rets]),
        "attachments": {},
    }
    if space["fmin"]["params"]["wandb_enable"]:
        wandb.log(
            {
                "accuracy_score": results["metric_classifier_accuracy"],
                "accuracy_score_std": results["metric_classifier_accuracy_std"],
                "precision": results["metric_classifier_precision_weighted"],
                "recall": results["metric_classifier_recall_weighted"],
                "f1_score": results["metric_classifier_fscore_weighted"],
                "f1_score_std": results["metric_classifier_fscore_weighted_std"],
                "accuracy_scores": [x["metric_classifier_accuracy"] for x in rets],
                "precisions": [x["metric_classifier_precision_weighted"] for x in rets],
                "recalls": [x["metric_classifier_recall_weighted"] for x in rets],
                "f1_scores": [x["metric_classifier_fscore_weighted"] for x in rets],
                "supports": [x["metric_classifier_support_weighted"] for x in rets],
                #                "support": ret['metric_classifier_support_weighted'],
                #                "val_precision_macro": ret['metric_classifier_precision_macro'],
                #                "val_recall_macro": ret['metric_classifier_recall_macro'],
                #                "val_fscore_macro": ret['metric_classifier_fscore_macro'],
                #                "val_support_macro": ret['metric_classifier_support_macro'],
                #                "val_precision_micro": ret['metric_classifier_precision_micro'],
                #                "val_recall_micro": ret['metric_classifier_recall_micro'],
                #                "val_fscore_micro": ret['metric_classifier_fscore_micro'],
                #                "val_support_micro": ret['metric_classifier_support_micro'],
            }
        )
    if space["fmin"]["params"]["wandb_enable"]:
        wandb.join()
    results["attachments"]["models"] = pickle.dumps([x["attachments"] for x in rets])
    return results


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


if __name__ == "__main__" and wandb._get_python_type() == "python":
    # # https://app.wandb.ai/jpizarrom/pan20apml/runs/3qlicko2/files/output.log

    parser = argparse.ArgumentParser()
    # parser.add_argument('--max_evals', type=int, default=2)
    parser.add_argument("--op", type=str, default="run_config")
    parser.add_argument("--conf", type=str, default="conf_0")
    parser.add_argument("--save-to", type=str)
    # parser.add_argument('--use_mongodb', action='store_true')

    args = parser.parse_args()
    print(args)

    if args.save_to is not None and os.path.exists(args.save_to):
        raise Exception("save_to exists")

    if args.op == "run_config":
        conf = importlib.import_module(args.conf)
        parameters_space = conf.parameters_space
    elif args.op == "run":
        run_path = args.conf
        print(run_path)
        api = wandb.Api()
        run = api.run(run_path)
        run_id = run.id
        parameters_space = {}
        config_defaults = dict(run.config)
        for key in config_defaults.keys():
            nkey = key
            if "fmin" in key and "fmin-name" not in key:
                nkey = key.replace("fmin-", "fmin-params-")
            if "feats" in key and "feats-name" not in key:
                nkey = key.replace("feats-", "feats-params-")
            if "classifier" in key and "classifier-name" not in key:
                nkey = key.replace("classifier-", "classifier-params-")
            nested_set(parameters_space, nkey.split("-"), config_defaults[key])
        print(json.dumps(parameters_space))
        if args.save_to:
            parameters_space["fmin"]["params"][
                "ds_name"
            ] = "pan20-author-profiling-training-2020-02-23"
            parameters_space["fmin"]["params"]["ds_name_folds"] = [""]
        parameters_space["classifier"]["params"]["max_iter"] = 5000
    else:
        raise NotImplemented()

    ret = objective_feats_and_clf(parameters_space)

    if args.save_to:
        print(args.save_to)
        models = pickle.loads(ret["attachments"]["models"])
        # vect = pickle.loads(models[0]["vect"])
        # clf = pickle.loads(models[0]["clf"])
        pipe = pickle.loads(models[0]["vect"])
        pipe.steps.append(["clf", pickle.loads(models[0]["clf"])])

        # vect_fname = "vectorizer.{}.pkl".format(args.save_to)
        # model_fname = "model.{}.pkl".format(args.save_to)
        os.makedirs(os.path.dirname(args.save_to), exist_ok=True)
        joblib.dump(pipe, args.save_to)
