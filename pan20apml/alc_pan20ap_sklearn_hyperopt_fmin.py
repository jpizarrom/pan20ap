#!/usr/bin/python
# coding: utf-8

import argparse
import importlib

# import os
import time
from hyperopt import fmin, tpe
from hyperopt import space_eval
from hyperopt import Trials
from alc_pan20ap_sklearn_hyperopt import objective_feats_and_clf


def _get_python_type():
    from IPython.core.getipython import get_ipython

    try:
        if "terminal" in get_ipython().__module__:
            return "ipython"
        else:
            return "jupyter"
    except (NameError, AttributeError):
        return "python"


# if _get_python_type() != "python":
#    !jupyter nbconvert --to python alc_pan20ap_sklearn_hyperopt.ipynb


# def objective_clf_dummy(space):
#     import os
#     print(os.getcwd())
#     results = {
#         'loss': 0,
#         'status': STATUS_OK,
#     }

parser = argparse.ArgumentParser()
parser.add_argument("--max_evals", type=int, default=2)
parser.add_argument("--conf", type=str, default="conf_0")
parser.add_argument("--use_mongodb", action="store_true")

args = parser.parse_args()
print(args)

max_evals = args.max_evals
use_mongodb = args.use_mongodb

conf = importlib.import_module(args.conf)
parameters_space = conf.parameters_space

run_suffix = parameters_space["fmin"]["params"]["run_suffix"]
lang = parameters_space["fmin"]["params"]["lang"]
task = parameters_space["fmin"]["params"]["task"]

start = time.time()
trials_filename = None  # 'trials_file'
if trials_filename:
    raise NotImplemented
#     try:
#         # https://github.com/PhilipMay/mltb/blob/427f9d7a56917090c9bf7e38474136770098156c/mltb/hyperopt.py#L80
#         trials = joblib.load(trials_filename)
#         evals_loaded_trials = len(trials.statuses())
# #        max_evals += evals_loaded_trials
#         print('{} evals loaded from trials file "{}".'.format(evals_loaded_trials, trials_filename))
#     except FileNotFoundError:
#         trials = Trials()
#         print('No trials file "{}" found. Created new trials object.'.format(trials_filename))
elif use_mongodb:
    from hyperopt.mongoexp import MongoTrials

    exp_key = "{}{}".format(lang, run_suffix)

    trials = MongoTrials("mongo://mongodb:27017/pan20apml/jobs", exp_key=exp_key)
else:
    trials = Trials()

best = fmin(
    objective_feats_and_clf,
    space=parameters_space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials,
    verbose=True,
)

if trials_filename:
    raise NotImplemented
#     joblib.dump(trials, trials_filename, compress=('gzip', 3))

print("Hyperopt search took %.2f seconds for xxx candidates" % ((time.time() - start)))
print(lang, task, best)
print(lang, task, space_eval(parameters_space, best)["classifier"])
print(lang, task, space_eval(parameters_space, best)["feats"])
