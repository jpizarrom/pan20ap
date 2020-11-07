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
            "ds_name_folds": [""],
            "hash_object": "46f2a631fc44535738a9ee3efa322d5a",
            "lang": "en",
            "model_suffix": ".alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a",
            "read_xmls": ".read_xmls_v0",
            "run_suffix": "alc_pan20ap_sklearn_hyperopt_fmin_conf_exp25",
            "task": "label",
            "wandb_enable": False,
            "wandb_save_model_enable": False,
            "xmls_base_directory": "../../pan20ap-ds",
        },
    },
}
