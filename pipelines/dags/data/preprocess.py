import pandas as pd
import numpy as np
import os, json, yaml, glob, shutil
import cv2
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tqdm import tqdm
from .validate import parallel_laplacian_variance, parallel_compare_images


def preprocess(param_file_name="params/param.yaml"):
    try:
        # parameter
        with open(param_file_name, "r") as config_file:
            params = yaml.safe_load(config_file)
        print("Parameters:", params)

        # log of eda file
        with open(os.path.join(params['log_folder'], f"eda_log_{params['version']}.json"), 'r', encoding='utf-8') as f:
            log_eda = json.load(f)

        print("EDA log:", log_eda)

        # file list
        files = sorted([file for file in glob.glob(f"{log_eda['files']['dir']}/*{params['extention']}")])
        print("Total files (Before removing):", len(files))

        # log file for preprocessing
        log_preprocess = {}


        # blurry image remove
        removed_blurry_files = []
        for i, file in tqdm(enumerate(files), total=len(files), desc="Blur: "):
            if parallel_laplacian_variance(file) < log_eda['blur']['blur_cutoff']:
                removed_blurry_files.append(file)
        blur_ratio = len(removed_blurry_files)/len(files)
        print("Blur ratio:", blur_ratio)
        log_preprocess['blur'] = {
            'ratio': blur_ratio,
            'num_files': len(removed_blurry_files),
            'total': len(files),
            'delete_blur_image' : params['delete_blur_image']
        }

        # duplicate image remove
        removed_duplicate_files = []
        diff_cutoff = log_eda['duplicate']['diff_cutoff']

        if diff_cutoff < 0.95:
            diff_cutoff = 0.95

        for i in tqdm(range(len(files)-1), total=len(files)-1, desc="Duplicate: "):
            if parallel_compare_images(i, files) < diff_cutoff:
                removed_duplicate_files.append(files[i])
        duplicate_ratio = len(removed_duplicate_files)/len(files)
        print("Duplicate ratio:", duplicate_ratio)
        log_preprocess['duplicate'] = {
            'ratio': duplicate_ratio,
            'num_files': len(removed_duplicate_files),
            'total': len(files),
            'delete_duplicate_image' : params['delete_duplicate_image'],
            'diff_cutoff': diff_cutoff
        }

        # train-val split
        if params['delete_blur_image']:
            files = list(set(files) - set(removed_blurry_files))
        if params['delete_duplicate_image']:
            files = list(set(files) - set(removed_blurry_files))

        print("Total files (After removing):", len(files))
        log_preprocess['preprocessed'] = {
            'num_files': len(files)
        }

        labels = []
        class_names = {name: 0 for name in params['class_names']}

        for file in files:
            for name in class_names:
                if name in file:
                    labels.append(name)
                    # class_names[name] = class_names[name]+1

        sss = StratifiedShuffleSplit(n_splits=1, test_size=params['test_ratio'], random_state=params['seed'])
        X_train, X_test, y_train, y_test = None, None, None, None
        for train_index, test_index in sss.split(files, labels):
            X_train, X_test = np.array(files)[train_index], np.array(files)[test_index]
            y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]


        # validate the distribution of class
        dist = {'train': {name: 0 for name in params['class_names']}, 'test': {name: 0 for name in params['class_names']}}

        for label in y_train:
            for name in class_names:
                if name in label:
                    dist['train'][name] = dist['train'][name]+1

        for label in y_test:
            for name in class_names:
                if name in label:
                    dist['test'][name] = dist['test'][name]+1

        print("Class distribution (after split):", dist)
        log_preprocess['preprocessed']['dist'] = dist

        # store preprocessed data in data warehouse 

        df = pd.DataFrame(columns = ['path', 'label', 'is_train'])
        train_folder = os.path.join(params['warehouse_dir'], params['version'], 'train')
        val_folder = os.path.join(params['warehouse_dir'], params['version'], 'val')
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        log_preprocess['preprocessed']['train_folder'] = train_folder
        log_preprocess['preprocessed']['val_folder'] = val_folder

        for i in range(len(X_train)):
            path = os.path.relpath(X_train[i], params['lake_dir'])
            if not os.path.exists(os.path.join(train_folder, path)):
                shutil.copyfile(X_train[i], os.path.join(train_folder, path))
            df = df.append({'path' : path, 'label' : y_train[i], 'is_train' : True}, ignore_index = True)

        for i in range(len(X_test)):
            path = os.path.relpath(X_test[i], params['lake_dir'])
            if not os.path.exists(os.path.join(val_folder, path)):
                shutil.copyfile(X_train[i], os.path.join(val_folder, path))
            df = df.append({'path' : path, 'label' : y_test[i], 'is_train' : False}, ignore_index = True)

        df.to_csv(os.path.join(params['warehouse_dir'], params['version'], 'data.csv'), index=False)
        log_preprocess['preprocessed']['data_file'] = os.path.join(params['warehouse_dir'], params['version'], 'data.csv')

        print('Log:', log_preprocess)

        with open(os.path.join(params['log_folder'], f"preprocess_log_{params['version']}.json"), 'w', encoding='utf-8') as f:
            json.dump(log_preprocess, f, ensure_ascii=False, indent=4)
        
        print("Completed")
    except Exception as e:
        print(e)
        print("Failed")







