import argparse
import glob
import os

import numpy as np
import scipy.io as sio
import torch
from torchsummary import summary
from tqdm import tqdm

from AutoGPU import autoGPU
from compare_model import SAE_3DCNN
from models import (_1DCNN, _2DCNN, _3DCNN, _3DCNN_1DCNN, _3DCNN_AM, PURE3DCNN,
                    PURE3DCNN_2AM, SAE, SAE_AM, DBDA_network, HamidaEtAl,
                    LeeEtAl)
from training_utils import TrainProcess
from utils import DataPreProcess, MyDataset, plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default='./bloodcell2-2/Split/5/', metavar='D',
                        help='the dataset path you load')
    parser.add_argument('--train_number', type=int,  default=2000, metavar='NTr', 
                        help='number of training set')
    parser.add_argument('--valid_number', type=int,  default=2000, metavar='NVa',
                        help='number of valid set')
    parser.add_argument('--test_number', type=int,  default=-1, metavar='NTe',
                        help='number of test set')
    parser.add_argument('--patchsize', type=int,  default=11, metavar='P',
                        help='patchsize of data')
    parser.add_argument('--modelname', type=str,  default='PURE3DCNN', metavar='P', help='which model to choose') 
    parser.add_argument('--gpu_ids', type=int,  default=5, metavar='G',
                        help='which gpu to use')
    parser.add_argument('--kw', type=int,  default=4, metavar='kw', help='MFA kw')
    parser.add_argument('--kb', type=int,  default=26, metavar='kb', help='MFA kb')
    parser.add_argument('--dim', type=int,  default=33, metavar='dim', help='MFA dim')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    args = parser.parse_args()
    gpu_ids = args.gpu_ids
    if gpu_ids != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
    else:
        autoGPU(1, 11000)

    dataset = args.dataset
    NTr = args.train_number
    NTe = args.test_number
    NVa = args.valid_number
    kw = args.kw
    kb = args.kb
    dim = args.dim
    patchsize = args.patchsize
    modelname = args.modelname
    if NTe == -1:
        d = dataset + 'TrNumber_%s/' % NTr + 'VaNumber_%s/' % NVa + 'TeNumber_all/'
    else:
        d = dataset + 'TrNumber_%s/' % NTr + 'VaNumber_%s/' % NVa + 'TeNumber_%s/' % NTe

    if not os.path.exists(d):
        os.makedirs(d)

    IMAGE = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell2-2.mat')['image']

    reduction_matrix = None if dim==IMAGE.shape[2] else sio.loadmat(d + 'Dim%s_Kw%s_Kb%s.mat'%(dim, kw, kb))['MFA_eigvector']
    testdata = DataPreProcess(IMAGE, patchsize,  d, 'Te', reduction_matrix=reduction_matrix, tasknum=20)
    traindata = DataPreProcess(IMAGE, patchsize, d, 'Tr', reduction_matrix=reduction_matrix, tasknum=20)
    validdata = DataPreProcess(IMAGE, patchsize, d, 'Va', reduction_matrix=reduction_matrix, tasknum=20)

    # assert trainpatch.shape[0] == 3*NTr, '预处理弄错了'
    assert traindata.patch.shape[3] == dim, '预处理弄错了'

    data_mix = {'train_patch': np.expand_dims(traindata.patch.transpose(0, 3, 1, 2), axis=1),
                'train_gt': traindata.gt,
                'test_patch': np.expand_dims(testdata.patch.transpose(0, 3, 1, 2), axis=1), 
                'test_gt': testdata.gt,
                'valid_patch': np.expand_dims(validdata.patch.transpose(0, 3, 1, 2), axis=1), 
                'valid_gt': validdata.gt,
                }

    models = {
              'SAE_3DCNN': SAE_3DCNN,
              '3DCNN': _3DCNN,
              '2DCNN': _2DCNN,
              'Hami': HamidaEtAl,
              '3DCNN_AM': _3DCNN_AM,
              'SAE': SAE,
              'Lee': LeeEtAl,
              'PURE3DCNN': PURE3DCNN,
              'DBDA': DBDA_network,
              'PURE3DCNN_2AM': PURE3DCNN_2AM,
              '1DCNN': _1DCNN,
              'DUALPATH': _3DCNN_1DCNN}

    model = models[modelname]
    themodel = model(dim).to('cuda')
    T = TrainProcess(model=themodel,
                     mixdata=data_mix, train_config='./config.yaml')
    T.training_start()
    name = '' if dim==IMAGE.shape[2] else 'mfa_Dim%s_kw%s_kb%s_' % (dim, kw, kb)
    np.save(d  + name + '%s_ConfMat.npy' % modelname, T.test_result.conf_mat)

    imgPos = np.array(list(traindata.pos) + list(validdata.pos) + list(testdata.pos))
    imgGt = np.array(list(traindata.gt) + list(validdata.gt) + T.test_result.y_pre)
    
    plot(imgPos, imgGt, IMAGE.shape, d + modelname)
