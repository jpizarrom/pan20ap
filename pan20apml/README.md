
# create kfolds datasets
folder: ../../pan20ap-ds

## 10kfold
pan20-author-profiling-training-2020-02-23-k10-0-dev
pan20-author-profiling-training-2020-02-23-k10-0-train
pan20-author-profiling-training-2020-02-23-k10-1-dev
pan20-author-profiling-training-2020-02-23-k10-1-train
pan20-author-profiling-training-2020-02-23-k10-2-dev
pan20-author-profiling-training-2020-02-23-k10-2-train
pan20-author-profiling-training-2020-02-23-k10-3-dev
pan20-author-profiling-training-2020-02-23-k10-3-train
pan20-author-profiling-training-2020-02-23-k10-4-dev
pan20-author-profiling-training-2020-02-23-k10-4-train
pan20-author-profiling-training-2020-02-23-k10-5-dev
pan20-author-profiling-training-2020-02-23-k10-5-train
pan20-author-profiling-training-2020-02-23-k10-6-dev
pan20-author-profiling-training-2020-02-23-k10-6-train
pan20-author-profiling-training-2020-02-23-k10-7-dev
pan20-author-profiling-training-2020-02-23-k10-7-train
pan20-author-profiling-training-2020-02-23-k10-8-dev
pan20-author-profiling-training-2020-02-23-k10-8-train
pan20-author-profiling-training-2020-02-23-k10-9-dev
pan20-author-profiling-training-2020-02-23-k10-9-train

## 5kfold
pan20-author-profiling-training-2020-02-23-k5-0-dev
pan20-author-profiling-training-2020-02-23-k5-0-train
pan20-author-profiling-training-2020-02-23-k5-1-dev
pan20-author-profiling-training-2020-02-23-k5-1-train
pan20-author-profiling-training-2020-02-23-k5-2-dev
pan20-author-profiling-training-2020-02-23-k5-2-train
pan20-author-profiling-training-2020-02-23-k5-3-dev
pan20-author-profiling-training-2020-02-23-k5-3-train
pan20-author-profiling-training-2020-02-23-k5-4-dev
pan20-author-profiling-training-2020-02-23-k5-4-train

# run
python -u alc_pan20ap_sklearn_hyperopt_fmin.py --conf alc_pan20ap_sklearn_hyperopt_fmin_conf_best_en_read_xmls_v0_k10_v0_kf --max_evals 1