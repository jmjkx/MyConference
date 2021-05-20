import argparse
import glob
import os
import pickle

import numpy as np
import scipy.io as sio
from sklearn import model_selection, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.random import sample_without_replacement

from old_utils import build_data
from utils import DataPreProcess, setpath

SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
                                       'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                   {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default='bloodcell1-3', metavar='D',
                        help='the dataset path you load')
    parser.add_argument('--trial_number', type=int,  default=1, metavar='NTr', 
                        help='number of training set')
    parser.add_argument('--train_number', type=int,  default=80, metavar='NTr', 
                        help='number of training set')
    parser.add_argument('--valid_number', type=int,  default=80, metavar='NVa',
                        help='number of valid set')
    parser.add_argument('--test_number', type=int,  default=2000, metavar='NTe',
                        help='number of test set')
    parser.add_argument('--patchsize', type=int,  default=11, metavar='P',
                        help='patchsize of data')
    parser.add_argument('--gpu_ids', type=int,  default=7, metavar='G',
                        help='which gpu to use')
    parser.add_argument('--kw', type=int,  default=9, metavar='kw', help='MFA kw')
    parser.add_argument('--kb', type=int,  default=8, metavar='kb', help='MFA kb')
    parser.add_argument('--dim', type=int,  default=33, metavar='dim', help='MFA dim')
    args = parser.parse_args()
    dataset = args.dataset
    NTr = args.train_number
    NVa = args.valid_number
    NTe = args.test_number
    trialnumber = args.trial_number
    NTe = args.test_number
    NVa = args.valid_number
    kw = args.kw
    kb = args.kb
    dim = args.dim
    patchsize = args.patchsize
    
    IMAGE = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell1-3.mat')['image']
    resultpath, imagepath, datapath = setpath(dataset, trialnumber , NTr, NVa, NTe, 'tradition')
    svmscore = []
    nnscore = []
    for i in range(1, 11):
        resultpath, imagepath, datapath = setpath(dataset, i , NTr, NVa, NTe, 'tradition')
     
        
        processeddata = DataPreProcess(IMAGE, patchsize, datapath).processeddata
        x_train = processeddata['train'].patch
        x_valid = processeddata['valid'].patch
        x_test = processeddata['test'].patch
        x_train = x_train[:, 5, 5, :]
        x_valid = x_valid[:, 5, 5, :]
        x_test = x_test[:, 5, 5, :]
        y_train = processeddata['train'].gt
        y_valid = processeddata['valid'].gt
        y_test = processeddata['test'].gt
        x_train = np.vstack((x_train, x_valid))
        y_train = np.hstack((y_train, y_valid))
        print("Running a grid search SVM")
    # Grid search SVM (linear and RBF)
        class_weight = 'balanced'
        
        # samplepos = [3,6,9]
        # x_train = x_train[samplepos]
       
        # y_train = y_train[samplepos]
      
        try:
            clf = svm.SVC(class_weight=class_weight)
            clf = model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4)
            clf.fit(x_train, y_train)
            print("SVM best parameters : {}".format(clf.best_params_))
        except:
            clf = svm.SVC(class_weight=class_weight)
            clf.fit(x_train, y_train)

        
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
    print('')
