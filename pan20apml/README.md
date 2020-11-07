
# pan20ap dataset
```
ls ../../pan20ap-ds/
pan20-author-profiling-training-2020-02-23
```

# create kfolds datasets
dataset folder: ../../pan20ap-ds

## create truth kfold
```
python alc_pan20ap_author_profiling_training_2019_02_18_get_ds_truth_kfold.py
```

output
```
0 ../../pan20ap-ds/pan20-author-profiling-training-2020-02-23/es/truth-k10-0-train.txt
0 ../../pan20ap-ds/pan20-author-profiling-training-2020-02-23/es/truth-k10-0-dev.txt
1 ../../pan20ap-ds/pan20-author-profiling-training-2020-02-23/es/truth-k10-1-train.txt
1 ../../pan20ap-ds/pan20-author-profiling-training-2020-02-23/es/truth-k10-1-dev.txt
...
```

## create kfold folders
```
python alc_pan20ap_author_profiling_training_2019_02_18_get_ds_kfold_folders.py
```

output
```
ls ../../pan20ap-ds/ -w1
...
pan20-author-profiling-training-2020-02-23-k10-0-dev
pan20-author-profiling-training-2020-02-23-k10-0-train
pan20-author-profiling-training-2020-02-23-k10-1-dev
pan20-author-profiling-training-2020-02-23-k10-1-train
...
```

# create df using read_xmls_v0, read_xmls_v1, read_xmls_v2
```
mkdir -p out
python alc_pan20ap_data_read.py
```

output samples
```
out/df_dev.en.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-0.read_xmls_v0.pkl
out/df_dev.en.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-0.read_xmls_v1.pkl
out/df_dev.en.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-0.read_xmls_v2.pkl
out/df_dev.en.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-1.read_xmls_v0.pkl
out/df_dev.en.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-1.read_xmls_v1.pkl
out/df_dev.en.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-1.read_xmls_v2.pkl
...
out/df_train.es.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-8.read_xmls_v0.pkl
out/df_train.es.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-8.read_xmls_v1.pkl
out/df_train.es.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-8.read_xmls_v2.pkl
out/df_train.es.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-9.read_xmls_v0.pkl
out/df_train.es.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-9.read_xmls_v1.pkl
out/df_train.es.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-9.read_xmls_v2.pkl
out/df_train.es.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a.read_xmls_v0.pkl
out/df_train.es.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a.read_xmls_v1.pkl
out/df_train.es.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a.read_xmls_v2.pkl
```

# create preprocessed df using preprocess_tweet_vX_X
```
python alc_pan20ap_data_preprocess_tweet.py
```

output
```
out/df_train.en.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-0.read_xmls_v0.preprocess_tweet_v0_0.pkl
out/df_train.en.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-1.read_xmls_v0.preprocess_tweet_v0_0.pkl
out/df_train.en.alc-sklearn-hyperopt.46f2a631fc44535738a9ee3efa322d5a-k10-2.read_xmls_v0.preprocess_tweet_v0_0.pkl
...
```

# run a conf
```
python -u alc_pan20ap_sklearn_hyperopt_fmin.py --conf alc_pan20ap_sklearn_hyperopt_fmin_conf_best_es_read_xmls_v0_k10_v1_kf_sample --max_evals 1
```

# run an experiment fold k10-0 
```
CFG_ds_name_folds=-k10-0 python -u alc_pan20ap_sklearn_hyperopt_fmin.py --conf alc_pan20ap_sklearn_hyperopt_fmin_conf_exp28 --max_evals 10
```

# run an experiment folds k10-0 to fold k10-9
```
CFG_ds_name_folds=-k10-0,-k10-1,-k10-2,-k10-3,-k10-4,-k10-5,-k10-6,-k10-7,-k10-8,-k10-9 python -u alc_pan20ap_sklearn_hyperopt_fmin.py --conf alc_pan20ap_sklearn_hyperopt_fmin_conf_exp28 --max_evals 1
```
