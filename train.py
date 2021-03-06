import argparse
import os
import pickle
import warnings

import numpy as np
import scipy.io as sio

from AutoGPU import autoGPU
from compare_model import SAE_3DCNN
from models import (_1DCNN, _2DCNN, _3DCNN, _3DCNN_1DCNN, _3DCNN_AM, PURE3DCNN,
                    PURE3DCNN_2AM, SAE, SAE_AM, DBDA_network, HamidaEtAl,
                    LeeEtAl, SSRN_network, myknn, mysvm)
from training_utils import TrainProcess, setup_seed
from utils import DataPreProcess, myplot, plot, setpath, splitdata

if __name__ == '__main__':
    setup_seed(1993)
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default='bloodcell1-3', metavar='D',
                        help='the dataset path you load')
    parser.add_argument('--trial_number', type=int,  default=1, metavar='T',
                        help='the time you do this trial')
    parser.add_argument('--train_number', type=int,  default=30, metavar='NTr', 
                        help='number of training set')
    parser.add_argument('--valid_number', type=int,  default=30, metavar='NVa',
                        help='number of valid set')
    parser.add_argument('--test_number', type=int,  default=-1, metavar='NTe',
                        help='number of test set')
    parser.add_argument('--patchsize', type=int,  default=11, metavar='P',
                        help='patchsize of data')
    parser.add_argument('--modelname', type=str,  default='SSRN', metavar='P', help='which model to choose') 
    parser.add_argument('--gpu_ids', type=int,  default=7, metavar='G',
                        help='which gpu to use')
    parser.add_argument('--kw', type=int,  default=9, metavar='kw', help='MFA kw')
    parser.add_argument('--kb', type=int,  default=8, metavar='kb', help='MFA kb')
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
    trialnumber = args.trial_number
    NTe = args.test_number
    NVa = args.valid_number
    kw = args.kw
    kb = args.kb
    dim = args.dim
    patchsize = args.patchsize
    modelname = args.modelname

    resultpath, imagepath, datapath = setpath(dataset, trialnumber , NTr, NVa, NTe, modelname)
    
    IMAGE = sio.loadmat('00microscope medical image data\sample1-3/bloodcell1-3.mat')['image']
    GND = sio.loadmat('00microscope medical image data\sample1-3/bloodcell1-3.mat')['map']
    spliteddata = splitdata(IMAGE, GND, datapath , trainnum=NTr, validnum=NVa, testnum=NTe)

    reduction_matrix = None if dim==IMAGE.shape[2] else sio.loadmat(resultpath + 'Dim%s_Kw%s_Kb%s.mat'%(dim, kw, kb))['MFA_eigvector']
    processeddata = DataPreProcess(IMAGE, patchsize, datapath, 5).processeddata

    data_mix = {'train_patch': np.expand_dims(processeddata['train'].patch.transpose(0, 3, 1, 2), axis=1),
                'train_gt': processeddata['train'].gt,
                'test_patch': np.expand_dims(processeddata['test'].patch.transpose(0, 3, 1, 2), axis=1), 
                'test_gt': processeddata['test'].gt,
                'valid_patch': None if processeddata['valid'] is None else np.expand_dims(processeddata['valid'].patch.transpose(0, 3, 1, 2), axis=1), 
                'valid_gt': None if processeddata['valid'] is None else processeddata['valid'].gt,
                }
    # data_mix = {'train_patch': np.expand_dims(np.load('A2S2K-ResNet/SSRN/x_train.npy'), axis=1),
    #             'train_gt':np.load('A2S2K-ResNet/SSRN/y_train.npy'),
    #             'test_patch': np.expand_dims(np.load('A2S2K-ResNet/SSRN/x_test.npy'), axis=1), 
    #             'test_gt': np.load('A2S2K-ResNet/SSRN/y_test.npy'),
    #             'valid_patch': np.expand_dims(np.load('A2S2K-ResNet/SSRN/x_val.npy'), axis=1), 
    #             'valid_gt': np.load('A2S2K-ResNet/SSRN/y_val.npy'),
    #             }

    assert data_mix['train_patch'].shape[0]==3*NTr, 'error'
    assert data_mix['test_patch'].shape[0]==3*NTe, 'error'
    assert data_mix['valid_patch'].shape[0]==3*NVa, 'error'

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
              'DUALPATH': _3DCNN_1DCNN,
              'SSRN':SSRN_network}

    model = models[modelname]
    themodel = model(dim).to('cuda')

    T = TrainProcess(model=themodel,
                     mixdata=data_mix,
                     train_config='./config.yaml',
                     writerpath=resultpath)
    T.training_start()

    name = '' if dim==IMAGE.shape[2] else 'mfa_Dim%s_kw%s_kb%s_' % (dim, kw, kb)
    
    with open(resultpath + 'result.pkl', 'wb') as f:
        pickle.dump(T.test_result, f, pickle.HIGHEST_PROTOCOL)
    myplot(processeddata, IMAGE, imagepath, T.test_result)
