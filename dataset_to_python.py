import scipy.io as sio
import os

'''
Matlab dataset to python for svm
'''

DATA_DIR = "./Data/"
DATA_TRAIN_NAME = "Product_Product"
DATA_TEST_NAME = "Product_Realworld"
METHOD_NAME = "SGF"

train_labels = sio.loadmat(os.path.join(DATA_DIR, DATA_TRAIN_NAME + ".mat"))["labels"]
train_features = sio.loadmat(os.path.join(DATA_DIR, METHOD_NAME, "train_feature.mat"))["train_feature"]

test_labels = sio.loadmat(os.path.join(DATA_DIR, DATA_TEST_NAME + ".mat"))["labels"]
test_features = sio.loadmat(os.path.join(DATA_DIR, METHOD_NAME, "test_feature.mat"))["test_feature"]

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

class SVM:
    def __init__(self, kernel="rbf"):
        self.kernel = kernel

    def train_and_test(self, train_X, train_y, test_X, test_y):
        print('Begin to train\nX dimention:\t{}'.format(train_X.shape[1]))
        svc = SVC(kernel=self.kernel, class_weight='balanced', )
        
        c_range = np.logspace(-2, 10, 12, base=2)
        gamma_range = ['auto']
        param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
        grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=4)
        
        clf = grid.fit(train_X, train_y)
        test_acc = grid.score(test_X, test_y)
        
        print("Testing accuracy: %f" %test_acc)

model = SVM()
model.train_and_test(train_features, train_labels, test_features, test_labels)