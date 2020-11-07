parameters_space = {
    "classifier": {
        "name": "LinearSVC",
        "params": {
            "C": 5968.385355812134,
            "class_weight": "balanced",
            "intercept_scaling": 1.4694306085857192,
            "loss": "hinge",
            "max_iter": 2000,
            "random_state": 42,
            "tol": 0.002814932892843107,
        },
    },
    "feats": {
        "name": "word_char",
        "params": {
            "char": {
                "max_df": 0.8,
                "min_df": 5,
                "ngram_range": (3, 5),
                "preprocessor": "preprocess_tweet_v2_0",
            },
            "word": {
                "max_df": 0.95,
                "min_df": 0.04,
                "ngram_range": (1, 2),
                "preprocessor": "preprocess_tweet_v3_1_1",
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
            "lang": "es",
            "model_suffix": ".alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a",
            "read_xmls": ".read_xmls_v0",
            "run_suffix": "alc_pan20ap_sklearn_hyperopt_fmin_conf_exp28",
            "task": "label",
            "wandb_enable": False,
            "wandb_save_model_enable": False,
            "xmls_base_directory": "../../pan20ap-ds",
        },
    },
}
