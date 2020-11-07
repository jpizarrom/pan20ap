import json
import glob
import hashlib
import math
import os
import re

# import neptune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from emoji import demojize
from xml.etree import ElementTree
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn import metrics
from tensorflow.keras.callbacks import History
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.keras.utils import plot_model

# from keras_mlflow import NeptuneMonitor
import wandb
from wandb.keras import WandbCallback


def read_xmls_v0(xmls_directory, truth_path=None):
    def _read_xml(author_id):
        tweets = []
        xml_filename = "{}.xml".format(author_id)
        tree = ElementTree.parse(
            os.path.join(xmls_directory, xml_filename),
            parser=ElementTree.XMLParser(encoding="utf-8"),
        )
        root = tree.getroot()
        for child in root[0]:
            tweets.append("xxnew {}".format(child.text))
        return " ".join(tweets)

    if truth_path is not None:
        df = pd.read_csv(
            truth_path, sep=":::", header=None, names=["author_id", "label"]
        )
    else:
        files = glob.glob("{}/*.xml".format(xmls_directory))
        author_ids = list(
            map(lambda x: os.path.splitext(os.path.basename(x))[0], files)
        )
        df = pd.DataFrame({"author_id": author_ids})

    df["tweet"] = df["author_id"].apply(_read_xml)
    return df


def read_xmls_v1(xmls_directory, truth_path=None):
    def _read_xml(author_id):
        tweets = []
        xml_filename = "{}.xml".format(author_id)
        tree = ElementTree.parse(
            os.path.join(xmls_directory, xml_filename),
            parser=ElementTree.XMLParser(encoding="utf-8"),
        )
        root = tree.getroot()
        for child in root[0]:
            tweets.append("{}".format(child.text))
        return " ".join(tweets)

    if truth_path is not None:
        df = pd.read_csv(
            truth_path, sep=":::", header=None, names=["author_id", "label"]
        )
    else:
        files = glob.glob("{}/*.xml".format(xmls_directory))
        author_ids = list(
            map(lambda x: os.path.splitext(os.path.basename(x))[0], files)
        )
        df = pd.DataFrame({"author_id": author_ids})

    df["tweet"] = df["author_id"].apply(_read_xml)
    return df


def read_xmls_v2(xmls_directory, truth_path=None):
    def _read_xml(author_id):
        tweets = []
        xml_filename = "{}.xml".format(author_id)
        tree = ElementTree.parse(
            os.path.join(xmls_directory, xml_filename),
            parser=ElementTree.XMLParser(encoding="utf-8"),
        )
        root = tree.getroot()
        for child in root[0]:
            tweets.append("{}".format(child.text))
        return tweets

    if truth_path is not None:
        df = pd.read_csv(
            truth_path, sep=":::", header=None, names=["author_id", "label"]
        )
    else:
        files = glob.glob("{}/*.xml".format(xmls_directory))
        author_ids = list(
            map(lambda x: os.path.splitext(os.path.basename(x))[0], files)
        )
        df = pd.DataFrame({"author_id": author_ids})

    df["tweet"] = df["author_id"].apply(_read_xml)
    df = df.explode("tweet")
    return df


read_xmls_funcs = {
    "read_xmls_v0": read_xmls_v0,
    "read_xmls_v1": read_xmls_v1,
    "read_xmls_v2": read_xmls_v2,
}


def write_file(filename, author_id, lang, atype):
    # <author id="author-id"
    #   lang="en|es"
    #   type="bot|human"
    #   gender="bot|male|female"
    # />
    tmpl = """
    <author id="{author_id}"
        lang="{lang}"
        type="{atype}"
    />"""
    value = tmpl.format(author_id=author_id, lang=lang, atype=atype,)
    with open(filename, "w") as f:
        f.write(value)


def load_df_train(
    xmls_directory_train,
    lang,
    model_suffix,
    truth_path_train,
    save_df=False,
    read_xmls=None,
):
    fname = "./out/df_train.{}{}.pkl".format(lang, model_suffix)
    print(fname)
    try:
        #    raise FileNotFoundError()
        df_train = pd.read_pickle(fname)
    except FileNotFoundError as e:
        if read_xmls is not None:
            print("use read_xmls ", read_xmls.__name__)
            df_train = read_xmls(xmls_directory_train, truth_path_train)
            if save_df and not os.path.exists(fname):
                print("to_pickle")
                df_train.to_pickle(fname)
        else:
            raise e

    print(df_train.shape)
    return df_train


def load_df_dev(
    xmls_directory_dev,
    lang,
    model_suffix,
    truth_path_dev,
    save_df=False,
    read_xmls=None,
):
    fname = "./out/df_dev.{}{}.pkl".format(lang, model_suffix)
    print(fname)
    try:
        #    raise FileNotFoundError()
        df_dev = pd.read_pickle(fname)
    except FileNotFoundError as e:
        if read_xmls is not None:
            print("use read_xmls ", read_xmls.__name__)
            df_dev = read_xmls(xmls_directory_dev, truth_path_dev)
            if save_df and not os.path.exists(fname):
                print("to_pickle")
                df_dev.to_pickle(fname)
        else:
            raise e

    print(df_dev.shape)
    return df_dev


def demojify(t, remove=True):
    t = demojize(t)
    if remove:
        return re.sub(r"(:[a-zA-Z0-9_-]+:)", " xxemj ", t)
    else:
        return re.sub(r"(:[a-zA-Z0-9_-]+:)", r" \1 ", t)


def replace_anonymized(t):
    t = t.replace("#URL#", "xxurl")
    t = t.replace("#USER#", "xxusr")
    t = t.replace("#HASHTAG#", "xxhst")
    return t


def replace_anonymized_v1(t):
    t = t.replace("#URL#…", "xxurl")
    t = t.replace("#USER#…", "xxusr")
    t = t.replace("#HASHTAG#…", "xxhst")

    t = t.replace("#URL#", "xxurl")
    t = t.replace("#USER#", "xxusr")
    t = t.replace("#HASHTAG#", "xxhst")
    return t


def replace_anonymized_v2(t):
    t = re.sub(r"(https?:?/{0,3}[a-z0-9.\-]*[…])", "xxurl", t)
    t = t.replace("#URL#…", "xxurl")
    t = t.replace("#USER#…", "xxusr")
    t = t.replace("#HASHTAG#…", "xxhst")

    t = t.replace("#URL#", "xxurl")
    t = t.replace("#USER#", "xxusr")
    t = t.replace("#HASHTAG#", "xxhst")
    return t


# use_pre_rules = True
# pre_rules = []
# if use_pre_rules:
#     pre_rules = [demojify]


def preprocess_tweet(
    tweet,
    pre_rules=[],
    preserve_case=True,
    reduce_len=True,
    replace_xxurl=False,
    replace_xxusr=False,
    replace_xxhst=False,
    replace_xxdgt=False,
):

    for pre_rule in pre_rules:
        tweet = pre_rule(tweet)

    # https://github.com/pan-webis-de/daneshvar18/blob/master/pan18ap/train_model.py#L146
    tokenizer = TweetTokenizer(preserve_case=preserve_case, reduce_len=reduce_len)
    tokens = tokenizer.tokenize(tweet)
    re_number = re.compile(r"^(?P<NUMBER>[+-]?\d+(?:[,/.:-]\d+[+-]?)?)$")
    for index, token in enumerate(tokens):
        if token[0:8] == "https://" or token[0:7] == "http://":
            if replace_xxurl:
                tokens[index] = "xxurl"
        elif token[0] == "@" and len(token) > 1:
            if replace_xxusr:
                tokens[index] = "xxusr"
        elif token[0] == "#" and len(token) > 1:
            if replace_xxhst:
                tokens[index] = "xxhst"
        elif re_number.match(token) is not None:
            if replace_xxdgt:
                tokens[index] = "xxdgt"
        elif token.isdigit():
            if replace_xxdgt:
                tokens[index] = "xxdgt"

    detokenizer = TreebankWordDetokenizer()
    tweet = detokenizer.detokenize(tokens)

    return tweet


def preprocess_tweet_v0_0(tweet):
    return preprocess_tweet(
        tweet,
        pre_rules=[],
        preserve_case=True,
        reduce_len=True,
        replace_xxurl=False,
        replace_xxusr=False,
        replace_xxhst=False,
        replace_xxdgt=False,
    )


def preprocess_tweet_v0_1(tweet):
    return preprocess_tweet(
        tweet,
        pre_rules=[],
        preserve_case=False,
        reduce_len=True,
        replace_xxurl=False,
        replace_xxusr=False,
        replace_xxhst=False,
        replace_xxdgt=False,
    )


def preprocess_tweet_v1_0(tweet):
    pre_rules = [demojify]
    return preprocess_tweet(
        tweet,
        pre_rules=pre_rules,
        preserve_case=True,
        reduce_len=True,
        replace_xxurl=False,
        replace_xxusr=False,
        replace_xxhst=False,
        replace_xxdgt=False,
    )


def preprocess_tweet_v1_1(tweet):
    pre_rules = [demojify]
    return preprocess_tweet(
        tweet,
        pre_rules=pre_rules,
        preserve_case=False,
        reduce_len=True,
        replace_xxurl=False,
        replace_xxusr=False,
        replace_xxhst=False,
        replace_xxdgt=False,
    )


def preprocess_tweet_v2_0(tweet):
    pre_rules = [replace_anonymized]
    return preprocess_tweet(
        tweet,
        pre_rules=pre_rules,
        preserve_case=True,
        reduce_len=True,
        replace_xxurl=False,
        replace_xxusr=False,
        replace_xxhst=False,
        replace_xxdgt=False,
    )


def preprocess_tweet_v2_0_1(tweet):
    pre_rules = [replace_anonymized]
    return preprocess_tweet(
        tweet,
        pre_rules=pre_rules,
        preserve_case=True,
        reduce_len=True,
        replace_xxurl=False,
        replace_xxusr=False,
        replace_xxhst=False,
        replace_xxdgt=True,
    )


def preprocess_tweet_v2_1(tweet):
    pre_rules = [replace_anonymized]
    return preprocess_tweet(
        tweet,
        pre_rules=pre_rules,
        preserve_case=False,
        reduce_len=True,
        replace_xxurl=False,
        replace_xxusr=False,
        replace_xxhst=False,
        replace_xxdgt=False,
    )


def preprocess_tweet_v2_1_1(tweet):
    pre_rules = [replace_anonymized]
    return preprocess_tweet(
        tweet,
        pre_rules=pre_rules,
        preserve_case=False,
        reduce_len=True,
        replace_xxurl=False,
        replace_xxusr=False,
        replace_xxhst=False,
        replace_xxdgt=True,
    )


def preprocess_tweet_v3_0(tweet):
    pre_rules = [replace_anonymized, demojify]
    return preprocess_tweet(
        tweet,
        pre_rules=pre_rules,
        preserve_case=True,
        reduce_len=True,
        replace_xxurl=False,
        replace_xxusr=False,
        replace_xxhst=False,
        replace_xxdgt=False,
    )


def preprocess_tweet_v3_0_1(tweet):
    pre_rules = [replace_anonymized, demojify]
    return preprocess_tweet(
        tweet,
        pre_rules=pre_rules,
        preserve_case=True,
        reduce_len=True,
        replace_xxurl=False,
        replace_xxusr=False,
        replace_xxhst=False,
        replace_xxdgt=True,
    )


def preprocess_tweet_v3_1(tweet):
    pre_rules = [replace_anonymized, demojify]
    return preprocess_tweet(
        tweet,
        pre_rules=pre_rules,
        preserve_case=False,
        reduce_len=True,
        replace_xxurl=False,
        replace_xxusr=False,
        replace_xxhst=False,
        replace_xxdgt=False,
    )


def preprocess_tweet_v3_1_1(tweet):
    pre_rules = [replace_anonymized, demojify]
    return preprocess_tweet(
        tweet,
        pre_rules=pre_rules,
        preserve_case=False,
        reduce_len=True,
        replace_xxurl=False,
        replace_xxusr=False,
        replace_xxhst=False,
        replace_xxdgt=True,
    )


preprocess_tweet_funcs = {
    "preprocess_tweet_v0_0": preprocess_tweet_v0_0,
    "preprocess_tweet_v0_1": preprocess_tweet_v0_1,
    "preprocess_tweet_v1_0": preprocess_tweet_v1_0,
    "preprocess_tweet_v1_1": preprocess_tweet_v1_1,
    "preprocess_tweet_v2_0": preprocess_tweet_v2_0,
    "preprocess_tweet_v2_0_1": preprocess_tweet_v2_0_1,
    "preprocess_tweet_v2_1": preprocess_tweet_v2_1,
    "preprocess_tweet_v2_1_1": preprocess_tweet_v2_1_1,
    "preprocess_tweet_v3_0": preprocess_tweet_v3_0,
    "preprocess_tweet_v3_0_1": preprocess_tweet_v3_0_1,
    "preprocess_tweet_v3_1": preprocess_tweet_v3_1,
    "preprocess_tweet_v3_1_1": preprocess_tweet_v3_1_1,
}


def load_glove_v0(word_index, max_features, embedding_file="./ds/glove.6B.50d.txt"):
    # https://www.kaggle.com/jpizarrom/gru-with-attention/edit
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype="float32")

    # embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file))
    # embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    print(emb_mean, emb_std)
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if word == "pronom_loc":
            embedding_vector = (
                np.ones(300) * 0.85
            )  # the number is arbitrary, just some vector
        if word == "person_loc":
            embedding_vector = (
                np.ones(300) * 0.25
            )  # the number is arbitrary, just some vector
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def load_glove_v1(
    tokenizer,
    word_index,
    max_features,
    use_mean=False,
    use_unknown_vector=False,
    embedding_file="/content/ds/glove.6B.50d.txt",
):
    if use_unknown_vector:
        raise
    # https://www.kaggle.com/wowfattie/3rd-place
    # EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype="float32")

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file))
    # embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    print(emb_mean, emb_std)
    embed_size = all_embs.shape[1]
    # embed_size = 50
    # nb_words = len(word_dict)+1
    unknown_words = []

    if use_mean:
        embedding_matrix = np.random.normal(
            emb_mean, emb_std, (max_features, embed_size)
        )
    else:
        embedding_matrix = np.zeros((max_features, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size + 1,), dtype=np.float32) - 1.0
    print(unknown_vector[:5])
    for key, i in word_index.items():
        if i >= max_features:
            continue
        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        # word = ps.stem(key)
        # embedding_vector = embeddings_index.get(word)
        # if embedding_vector is not None:
        #     embedding_matrix[i] = embedding_vector
        #     continue
        # word = lc.stem(key)
        # embedding_vector = embeddings_index.get(word)
        # if embedding_vector is not None:
        #     embedding_matrix[i] = embedding_vector
        #     continue
        # word = sb.stem(key)
        # embedding_vector = embeddings_index.get(word)
        # if embedding_vector is not None:
        #     embedding_matrix[i] = embedding_vector
        #     continue
        # word = lemma_dict[key]
        # embedding_vector = embeddings_index.get(word)
        # if embedding_vector is not None:
        #     embedding_matrix[i] = embedding_vector
        #     continue
        # if len(key) > 1:
        #     word = correction(key)
        #     embedding_vector = embeddings_index.get(word)
        #     if embedding_vector is not None:
        #         embedding_matrix[i] = embedding_vector
        #         continue
        unknown_words.append(word)
        if use_unknown_vector:
            embedding_matrix[i] = unknown_vector
    # return embedding_matrix, nb_words
    return embedding_matrix, unknown_words


def fit(
    build_model_fn,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=32,
    epochs=50,
    verbose=False,
    callbacks=None,
    **kwargs
):

    model = build_model_fn(**kwargs)
    return fit_model(
        build_model_fn,
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=32,
        epochs=50,
        verbose=False,
        callbacks=None,
        **kwargs
    )


def fit_model(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=32,
    epochs=50,
    verbose=False,
    callbacks=None,
    **kwargs
):

    cb = []
    history = History()
    cb.append(history)

    if callbacks:
        cb += callbacks

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=cb,
        verbose=verbose,
    )
    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=verbose)
    print("Training Accuracy: {:.4f}".format(train_accuracy))
    val_loss, val_accuracy = model.evaluate(x_test, y_test, verbose=verbose)
    print("Testing Accuracy:  {:.4f}".format(val_accuracy))
    # plot_history(history)
    return (train_loss, train_accuracy), (val_loss, val_accuracy), model, history


def plot_history(history):
    try:
        acc = history.history["acc"] if "acc" in history.history else None
        val_acc = history.history["val_acc"] if "val_acc" in history.history else None
        loss = history.history["loss"] if "loss" in history.history else None
        val_loss = (
            history.history["val_loss"] if "val_loss" in history.history else None
        )
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        if acc:
            plt.plot(x, acc, "b", label="Training acc")
        if val_acc:
            plt.plot(x, val_acc, "r", label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.subplot(1, 2, 2)
        if loss:
            plt.plot(x, loss, "b", label="Training loss")
        if val_loss:
            plt.plot(x, val_loss, "r", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
    except Exception as e:
        print(e)
    plt.show()


# os.environ['NEPTUNE_API_TOKEN'] = ''
# os.environ["NEPTUNE_PROJECT"] = ''


def fit_and_netptune(
    build_model_fn,
    x_train,
    y_train,
    x_test,
    y_test,
    embeddings=None,
    batch_size=64,
    epochs=50,
    verbose=False,
    callbacks=None,
    use_neptune=False,
    use_wandb=True,
    run_meta_summary={},
    run_meta_config={},
    space={},
):

    if use_neptune:
        try:
            project = neptune.init()
        except:
            pass
    # params = {}

    # for k in run_meta_config.keys():
    #     params[k] = run_meta_config[k]

    # for _, row in pd.io.json.json_normalize({'data':space['data']}).iterrows():
    #     for k in row.keys():
    #         params[k] = row[k]

    # for _, row in pd.io.json.json_normalize({'build_model':space['build_model']}).iterrows():
    #     for k in row.keys():
    #         params[k] = row[k]

    # for _, row in pd.io.json.json_normalize({'fit':space['fit']}).iterrows():
    #     for k in row.keys():
    #         params[k] = row[k]

    # for _, row in pd.io.json.json_normalize(kwargs).iterrows():
    #     for k in row.keys():
    #         params[k] = row[k]

    import copy

    build_model_fn_kwargs = copy.deepcopy(space["build_model"]["args"])
    build_model_fn_kwargs["embedding_matrix_weights"] = embeddings[
        space["data"]["embedding_matrix_weights"]
    ]

    model = build_model_fn(**build_model_fn_kwargs)
    model.summary()

    total_count = model.count_params()
    non_trainable_count = count_params(model.non_trainable_weights)
    trainable_count = total_count - non_trainable_count

    # params['build_model_fn'] = build_model_fn.__name__
    # params['batch_size'] = batch_size
    # params['epochs'] = epochs

    # if use_wandb:
    #     wandb.init(project="pan20apdl", config=space, reinit=True, allow_val_change=True)
    if use_wandb:
        wandb.run.summary["build_model_hash_kwargs"] = hashlib.md5(
            json.dumps(
                {
                    "build_model_fn": build_model_fn.__name__,
                    "args": space["build_model"]["args"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        wandb.run.summary["build_model_hash"] = hashlib.md5(
            json.dumps(space["build_model"], sort_keys=True).encode()
        ).hexdigest()

        # print('build_model_hash_kwargs', wandb.run.summary['build_model_hash_kwargs'])
        # print('build_model_hash', wandb.run.summary['build_model_hash'])

        # print('params_total_count', total_count)
        # print('params_non_trainable_count', non_trainable_count)
        # print('params_trainable_count', trainable_count)

        wandb.run.summary["params_total_count"] = total_count
        wandb.run.summary["params_non_trainable_count"] = non_trainable_count
        wandb.run.summary["params_trainable_count"] = trainable_count

        for k in run_meta_summary.keys():
            wandb.run.summary[k] = run_meta_summary[k]

    if use_neptune:
        neptune_aborted = None

        def stop_training():
            nonlocal neptune_aborted
            neptune_aborted = True
            model.stop_training = True

        tags = []
        # tags = [params['build_model_fn']]
        try:
            project.create_experiment(
                name="runner_qualified_name",
                params=params,
                upload_source_files=[],
                abort_callback=stop_training,
                tags=tags,
            )
        except:
            pass
    # if use_wandb:
    #     wandb.config.update(params, allow_val_change=True)
    # for k, v in params.items():
    #     setattr(wandb.config, k, v)
    print(space)
    cb = []
    history = History()
    cb.append(history)

    if use_neptune:
        cb.append(NeptuneMonitor())

    if use_wandb:
        cb.append(
            WandbCallback(
                save_model=False,
                validation_data=(x_test, y_test),
                monitor="val_accuracy",
            )
        )

    if callbacks:
        cb += callbacks

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=cb,
        verbose=verbose,
    )

    if use_neptune:
        print("neptune_aborted", neptune_aborted)
        if not neptune_aborted:
            try:
                neptune.stop()
            except:
                pass

    # loss_tr, accuracy_tr = model.evaluate(x_train, y_train, verbose=verbose)
    # print("Training Accuracy: {:.4f}".format(accuracy_tr))
    # loss_te, accuracy_te = model.evaluate(x_test, y_test, verbose=verbose)
    # print(loss_te, accuracy_te)

    # y_pred = model.predict(x_test)
    # y_pred = np.floor(y_pred * 2)

    y_pred = model.predict_classes(x_test, verbose=1)
    y_pred = np.squeeze(y_pred)

    accuracy_te = metrics.accuracy_score(y_test, y_pred)
    (
        precisions_weighted,
        recalls_weighted,
        f_measures_weighted,
        support_weighted,
    ) = metrics.precision_recall_fscore_support(y_test, y_pred, average="weighted")
    print("Testing Accuracy:  {:.4f}".format(accuracy_te))
    # plot_history(history)
    # if wandb:
    #     wandb.join()
    return (
        model,
        (None, accuracy_te),
        (precisions_weighted, recalls_weighted, f_measures_weighted, support_weighted),
    )
