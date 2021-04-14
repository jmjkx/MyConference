import argparse
import glob
import os

import numpy as np
import scipy.io as sio

from models import myknn, mysvm
from utils import DataPreProcess, plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default='./bloodcell1-3/Split/5/', metavar='D',
                        help='the dataset path you load')
    parser.add_argument('--train_number', type=int,  default=80, metavar='NTr',
                        help='number of training set')
    parser.add_argument('--valid_number', type=int,  default=80, metavar='NVa',
                        help='number of valid set')
    parser.add_argument('--test_number', type=int,  default=-1, metavar='NTe',
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
                        '高光谱医学/高光谱医学数据/bloodcell1-3.mat')['image']
  
    if NTe != -1:
        d = dataset + 'TrNumber_%s/' % NTr + 'VaNumber_%s/' % NVa + 'TeNumber_%s/' % NTe
    else:
        d = dataset + 'TrNumber_%s/' % NTr + 'VaNumber_%s/' % NVa + 'TeNumber_all/' 
    testdata = DataPreProcess(IMAGE, 1,  d, 'Te', reduction_matrix=None, tasknum=20)
    traindata = DataPreProcess(IMAGE, 1, d, 'Tr', reduction_matrix=None, tasknum=20)
    validdata = DataPreProcess(IMAGE, 1, d, 'Va', reduction_matrix=None, tasknum=20)
    x_train = traindata.patch[:, 0, 0, :]
    x_valid = validdata.patch[:, 0, 0, :]
    x_test = testdata.patch[:, 0, 0, :]
    y_train = np.array(traindata.gt)
    y_valid = np.array(validdata.gt)
    y_test = np.array(testdata.gt)
    x_train = np.vstack((x_train, x_valid))
    y_train = np.hstack((y_train, y_valid))
    print("Running a grid search SVM")
# Grid search SVM (linear and RBF)
    svm_prediction = mysvm(x_train, y_train, x_test)
    knn_prediction = myknn(x_train, y_train, x_test)
    imgPos = np.array(list(traindata.pos) + list(validdata.pos) + list(testdata.pos))
    imgGt = np.array(list(traindata.gt) + list(validdata.gt) + list(knn_prediction))
    plot(imgPos, imgGt, IMAGE.shape, d + 'NearesetNeighbor')
    imgGt = np.array(list(traindata.gt) + list(validdata.gt) + list(svm_prediction))
    plot(imgPos, imgGt, IMAGE.shape, d + 'SVM')

  
