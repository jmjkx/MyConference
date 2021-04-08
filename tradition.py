import argparse
import glob
import os

import numpy as np
import scipy.io as sio
from sklearn import model_selection, svm
from sklearn.neighbors import KNeighborsClassifier

from utils import build_data

SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
                                       'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                   {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default='./bloodcell2-2/Split/', metavar='D',
                        help='the dataset path you load')
    parser.add_argument('--train_number', type=int,  default=80, metavar='NTr',
                        help='number of training set')
    parser.add_argument('--valid_number', type=int,  default=80, metavar='NVa',
                        help='number of valid set')
    parser.add_argument('--test_number', type=int,  default=2000, metavar='NTe',
                        help='number of training set')
    parser.add_argument('--savepath', type=str,  default='./bloodcell2-2/result/1', metavar='S',
                        help='experiment result path to save')
    args = parser.parse_args()
    dataset = args.dataset
    NTr = args.train_number
    NVa = args.valid_number
    NTe = args.test_number
    savepath = args.savepath
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    IMAGE = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell2-2.mat')['image']
    svmscore = []
    nnscore = []
    for i in range(1, 11):
        d = dataset +  str(i) + '/' + 'TrNumber_%s/' % NTr + 'VaNumber_%s/' % NVa + 'TeNumber_%s/' % NTe
        x_train, y_train = build_data(IMAGE, 1, d, 'Tr')
        x_valid, y_valid = build_data(IMAGE, 1, d, 'Va')
        x_test, y_test = build_data(IMAGE, 1, d, 'Te')
        x_train = x_train[:, 0, 0, :]
        x_valid = x_valid[:, 0, 0, :]
        x_test = x_test[:, 0, 0, :]
        x_train = np.vstack((x_train, x_valid))
        y_train = np.hstack((y_train, y_valid))
        print("Running a grid search SVM")
    # Grid search SVM (linear and RBF)
        class_weight = 'balanced'
        clf = svm.SVC(class_weight=class_weight)
        clf = model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4)
        clf.fit(x_train, y_train)
        print("SVM best parameters : {}".format(clf.best_params_))
        prediction = clf.predict(x_test)
        svmscore.append(sum(prediction==y_test)/y_test.shape[0])
        kclf = KNeighborsClassifier(n_neighbors=1)
        kclf.fit(x_train, y_train)
        pre_test = kclf.predict(x_test)   
        nnscore.append(sum(pre_test==y_test)/y_test.shape[0])
    
    print('=============NN============')
    print('%.2f'%(np.mean(nnscore)*100) + '±' +'%.2f'%(np.std(nnscore, ddof=1)*100))
    print('=============SVM============')
    print('%.2f'%(np.mean(svmscore)*100) + '±' +'%.2f'%(np.std(svmscore, ddof=1)*100))
