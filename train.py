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
from utils import MyDataset, build_data, plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default='./bloodcell2-2/Split/5/', metavar='D',
                        help='the dataset path you load')
    parser.add_argument('--train_number', type=int,  default=80, metavar='NTr', help='number of training set') 
    parser.add_argument('--valid_number', type=int,  default=80, metavar='NTe',
                        help='number of training set')
    parser.add_argument('--test_number', type=int,  default=-1, metavar='NTe',
                        help='number of test set')
    parser.add_argument('--savepath', type=str,  default='./bloodcell2-2/result/1', metavar='S', help='experiment result path to save') 
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
    savepath = args.savepath
    modelname = args.modelname
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if NTe == -1:
        d = dataset + 'TrNumber_%s/' % NTr + 'VaNumber_%s/' % NVa + 'TeNumber_all/'
    else:
        d = dataset + 'TrNumber_%s/' % NTr + 'VaNumber_%s/' % NVa + 'TeNumber_%s/' % NTe 

    # 定义超参
    IMAGE = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell2-2.mat')['image']

    if dim != 33:
        reduction_matrix = sio.loadmat(d  + 'Dim%s_Kw%s_Kb%s.mat'%(dim, kw, kb))['MFA_eigvector']
                                   
    else:
        reduction_matrix = None
    testpatch, gt_test = build_data(IMAGE, patchsize,  d, 'Te', reduction_matrix=reduction_matrix)
    trainpatch, gt_train = build_data(IMAGE, patchsize, d, 'Tr', reduction_matrix=reduction_matrix)

    validpatch, gt_valid = build_data(IMAGE, patchsize, d, 'Va', reduction_matrix=reduction_matrix)

    # assert trainpatch.shape[0] == 3*NTr, '预处理弄错了'
    assert trainpatch.shape[3] == dim, '预处理弄错了'

    data_mix = {'train_patch': np.expand_dims(trainpatch.transpose(0, 3, 1, 2), axis=1),
                'train_gt': gt_train,
                'test_patch': np.expand_dims(testpatch.transpose(0, 3, 1, 2), axis=1), 
                'test_gt': gt_test,
                'valid_patch': np.expand_dims(validpatch.transpose(0, 3, 1, 2), axis=1), 
                'valid_gt': gt_valid,
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
    # trainpatch = trainpatch.squeeze()
    # testpatch = testpatch.squeeze()
    # data_mix = {'train_patch': trainpatch,
    #             'train_gt': gt_train,
    #             'test_patch': testpatch, 
    #             'test_gt': gt_test}

    model = models[modelname]
    themodel = model(dim).to('cuda')
    # summary(themodel, input_shape)
    T = TrainProcess(model=themodel,
                     mixdata=data_mix, train_config='./config.yaml')
    T.training_start()
    name = ''
    if dim != 33:
        name = 'mfa_Dim%s_kw%s_kb%s_' % (dim, kw, kb)
    np.save(d  + name + '%s_ConfMat.npy' % modelname, T.test_result.conf_mat)
    


    index_files = sorted(glob.glob(d +'XTeind_class*.npy'), key=lambda x: int(x[-5]))
    datasetPos = []
    for f in index_files:
        datasetPos += list(np.load(f))
    datasetPos = np.array(datasetPos)
    plot(datasetPos, T.test_result.y_pre, IMAGE.shape, './rrr')
