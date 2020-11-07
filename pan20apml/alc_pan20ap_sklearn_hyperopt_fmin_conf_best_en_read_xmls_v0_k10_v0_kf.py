import os

parameters_space = {
    "classifier": {
        "name": "LinearSVC",
        "params": {
            "C": 22458.780264488676,
            "class_weight": "balanced",
            "intercept_scaling": 0.8769049589825447,
            "loss": "hinge",
            "max_iter": 2000,
            "random_state": 42,
            "tol": 0.00014376187244711285,
        },
    },
    "feats": {
        "name": "word_char",
        "params": {
            "char": {
                "max_df": 1.0,
                "min_df": 0.001,
                "ngram_range": (1, 6),
                "preprocessor": "preprocess_tweet_v2_1_1",
            },
            "word": {
                "max_df": 0.85,
                "min_df": 0.05,
                "ngram_range": (1, 2),
                "preprocessor": "preprocess_tweet_v2_1_1",
                "stop_words": None,
            },
        },
    },
    "fmin": {
        "name": "fmin",
        "params": {
            "ds_name": "pan20-author-profiling-training-2020-02-23",
            "ds_name_folds": ["-k10-{}".format(i) for i in range(1)],
            "hash_object": "46f2a631fc44535738a9ee3efa322d5a",
            "lang": "en",
            "model_suffix": ".alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a",
            "read_xmls": ".read_xmls_v0",
            "run_suffix": os.path.splitext(os.path.basename(__file__))[0],
            "task": "label",
            "wandb_enable": True,
            "wandb_save_model_enable": False,
            "xmls_base_directory": "../../pan20ap-ds",
        },
    },
}
