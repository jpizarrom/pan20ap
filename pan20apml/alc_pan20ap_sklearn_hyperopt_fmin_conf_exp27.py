
import hashlib
import os
import numpy as np
from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials


run_suffix = os.path.splitext(os.path.basename(__file__))[0]
wandb_enable = os.environ.get('CFG_wandb_enable', 'False') == 'True'
wandb_save_model_enable = os.environ.get('CFG_wandb_save_model_enable', 'False') == 'True'

xmls_base_directory = '../../pan20ap-ds'

ds_name_train = os.environ.get('CFG_ds_name_train', 'pan20-author-profiling-training-2020-02-23-k5')

ds_name_folds = os.environ.get('CFG_ds_name_folds', '-0')
ds_name_folds = ds_name_folds.split(',')

read_xmls = os.environ.get('CFG_read_xmls', '.read_xmls_v0')

lang = os.environ.get('CFG_lang','en')
task = 'label'

hash_object = hashlib.md5(ds_name_train.encode()).hexdigest()

model_suffix = '.alc-sklearn-hyperopt'
model_suffix += '.{}'.format(hash_object)
print(lang, task, model_suffix, hash_object)

print('run_suffix', run_suffix, type(run_suffix))
print('wandb_enable', wandb_enable, type(wandb_enable))
print('wandb_save_model_enable', wandb_save_model_enable, type(wandb_save_model_enable))

print('xmls_base_directory', xmls_base_directory, type(xmls_base_directory))

print('ds_name_train', ds_name_train, type(ds_name_train))

print('lang', lang, type(lang))
print('task', task, type(task))
# print('train_model', train_model, type(train_model))
# print('max_evals', max_evals, type(max_evals))

print('hash_object', hash_object, type(hash_object))

print('ds_name_folds', ds_name_folds, type(ds_name_folds))

space_svm = {}
space_svm['C'] = hp.loguniform('C', np.log(1e-5), np.log(1e5))
space_svm['loss'] = hp.choice('svm_loss', [
    'hinge', 
    'squared_hinge',
])
space_svm['tol'] = hp.loguniform('tol', np.log(1e-5), np.log(1e-2))
space_svm['intercept_scaling'] = hp.loguniform('intercept_scaling', np.log(1e-1), np.log(1e1))
space_svm['class_weight'] = hp.choice('class_weight', [None, 'balanced'])
space_svm['max_iter'] = 2000
space_svm['random_state'] = 42

space_fmin = {
            'name': 'fmin',
            'params': {
                'run_suffix': run_suffix,
                'wandb_enable': wandb_enable,
                'wandb_save_model_enable': wandb_save_model_enable,
                'lang': lang,
                'task': 'label',
#                'xmls_base_directory_train_name': xmls_base_directory_train_name,
#                'xmls_base_directory_train': xmls_base_directory_train,
#                'xmls_base_directory_dev': xmls_base_directory_dev,
                'xmls_base_directory': xmls_base_directory,
                'ds_name': ds_name_train,
                'read_xmls': read_xmls,
                'ds_name_folds': hp.choice('read_xmls', [['{}{}'.format(ds_name_fold, read_xmls) for ds_name_fold in ds_name_folds]]),
#                'max_evals': max_evals,
                'hash_object': hash_object,
                'model_suffix': model_suffix,
            }
}

parameters_space = {
    'feats': hp.choice('feats', [
        {'name':'eda', 'params': {}},
    ]),
    'classifier': hp.choice('classifier', [
        {'name':'LinearSVC', 'params': space_svm},
    ]),
    'fmin': space_fmin,
}

# print(parameters_space)