import scipy.io as scio
import pandas as pd
import numpy as np
import os

DATA_DIR = "./Data/"

def make_dataset_mat(file_dir):
    save_name = file_dir.split(".")[0]
    file_dir = os.path.join(DATA_DIR, file_dir)
    data = pd.read_csv(file_dir).values

    features = data[:, :-1]
    labels = data[:, -1]
    labels = np.expand_dims(labels, axis=1)

    scio.savemat(os.path.join(DATA_DIR, save_name + ".mat"), {"features": features, "labels": labels})

for file_dir in os.listdir(DATA_DIR):
    make_dataset_mat(file_dir)