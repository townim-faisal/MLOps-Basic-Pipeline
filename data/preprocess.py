import pandas as pd
import os, json, yaml, glob
import cv2
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from validate import parallel_laplacian_variance, parallel_compare_images

# parameter
with open("params/param.yaml", "r") as config_file:
    params = yaml.safe_load(config_file)
print("Parameters:", params)

# log of eda file
with open(os.path.join(params['log_folder'], f"eda_log_{params['version']}.json"), 'r', encoding='utf-8') as f:
    log_eda = json.load(f)

print("EDA log:", log_eda)

# file list
files = sorted([file for file in glob.glob(f"{log_eda['files']['dir']}/*{params['extention']}")])
print("Total files:", len(files))

# log file for preprocessing
log_preprocess = {}


# blurry image remove
removed_blurry_files = []
for file in files:
    if parallel_laplacian_variance(file) < log_eda['blur']['blur_cutoff']:
        removed_blurry_files.append(file)
        # os.remove(files[i])
blur_ratio = len(removed_blurry_files)/len(files)
print("Blur ratio:", blur_ratio)

# duplicate image remove
removed_duplicate_files = []
# diff_cutoff = log_eda['duplicate']['diff_cutoff']

# if diff_cutoff < 0.95:
#     diff_cutoff = 0.95

# for i in range(len(files)-1):
#     if parallel_compare_images(i, files) < diff_cutoff:
#         removed_duplicate_files.append(files[i])
# duplicate_ratio = len(removed_duplicate_files)/len(files)
# print("Duplicate ratio:", duplicate_ratio)

# train-val split
main_files = list(set(files) - set(removed_blurry_files) - set(removed_duplicate_files))
print(len(main_files))
labels = []
class_names = {name: 0 for name in params['class_names']}

for file in main_files:
    for name in class_names:
        if name in file:
            labels.append(name)
            # class_names[name] = class_names[name]+1

# X_train, X_test, y_train, y_test = train_test_split(main_files, labels, test_size=params['test_ratio'], random_state=params['seed'])
sss = StratifiedShuffleSplit(n_splits=1, test_size=params['test_ratio'], random_state=params['seed'])
X_train, X_test, y_train, y_test = None, None, None, None
for train_index, test_index in sss.split(main_files, labels):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = main_files[train_index], main_files[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


# validate the distribution of class
dist = {'train': {name: 0 for name in params['class_names']}, 'test': {name: 0 for name in params['class_names']}}

for file in y_train:
    for name in class_names:
        if name in file:
            dist['train'][name] = dist['train'][name]+1

for file in y_test:
    for name in class_names:
        if name in file:
            dist['test'][name] = dist['test'][name]+1

print(dist)

