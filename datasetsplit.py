import argparse
import os
import time

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='bloodcell2-2',
                        metavar='D', help='dataset')
    parser.add_argument('--savepath', type=str, 
                        default='./bloodcell2-2/Split/5',
                        metavar='S', help='the path to save')
    parser.add_argument('--ntrain', type=int, metavar='NTr',
                        default=5, help='number of training samples in each class')
    parser.add_argument('--nvalid', type=int, metavar='NTe',
                        default=5, help='number of training samples in each class')
    parser.add_argument('--ntest', type=int, metavar='NTe',
                        default=2000, help='number of test samples in each class')
    parser.add_argument('--patchsize', type=int, metavar='PS',
                        default=11, help='the number of patchsize')
    args = parser.parse_args()
    dataset = args.dataset
    patchsize = args.patchsize
    savepath = args.savepath
    NTr = args.ntrain
    NVa = args.nvalid
    NTe = args.ntest
    datasetpath = dataset + '/Entire/'
    if NTe != -1:
        savepath = savepath + '/TrNumber_%s/' % NTr\
                + 'VaNumber_%s/' % NVa\
                + 'TeNumber_%s/'% NTe
    else:
        savepath = savepath + '/TrNumber_%s/' % NTr\
                + 'VaNumber_%s/' % NVa\
                + 'TeNumber_all/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for label in range(3):
        X_entire = np.load(datasetpath + 'imgind_class%s.npy'%label)
        Y_entire = np.load(datasetpath + 'gt_class%s.npy'%label)
        if NTe != -1:
            indice = sample_without_replacement(X_entire.shape[0], NTr+NVa+NTe) 
            X_entire = X_entire[indice]
            Y_entire = Y_entire[indice]
        x_trainind, x_tevaind,\
        y_trainind, y_tevaind = train_test_split(X_entire, Y_entire, 
                                                 test_size = (Y_entire.shape[0] - NTr ) / Y_entire.shape[0],
                                                 random_state = time.localtime(time.time()).tm_sec, 
                                                 stratify = Y_entire)

        x_validind, x_testind,\
        y_validind, y_testind = train_test_split(x_tevaind, y_tevaind, 
                                                 test_size = (y_tevaind.shape[0] - NVa)/ y_tevaind.shape[0],
                                                 random_state = time.localtime(time.time()).tm_sec, 
                                                 stratify = y_tevaind) 

        assert all([x_trainind.shape[0] == NTr,   
                    x_testind.shape[0] == NTe,
                    x_validind.shape[0] == NVa]), "错了"
        np.save(savepath+'XTrind_class%s' % label, x_trainind)
        np.save(savepath+'XTeind_class%s' % label, x_testind)
        np.save(savepath+'YTr_class%s' % label, y_trainind)
        np.save(savepath+'YTe_class%s' % label, y_testind)
        np.save(savepath+'XVaind_class%s' % label, x_validind)
        np.save(savepath+'YVa_class%s' % label, y_validind)
    print('==================end==================')
