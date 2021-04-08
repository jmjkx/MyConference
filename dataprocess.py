import argparse

import numpy as np
import scipy.io as sio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str, 
                        default='/home/liyuan/Programming/python/高光谱医学/高光谱医学数据/bloodcell2-2.mat', 
                        metavar='D',help='datasetpath')
    parser.add_argument('--savepath', type=str, 
                        default='./bloodcell2-2/Entire/', 
                        metavar='S',help='the path to save dataset')
    args = parser.parse_args()
    datasetpath = args.datasetpath
    savepath = args.savepath
    D = sio.loadmat(datasetpath)['image']
    T = sio.loadmat(datasetpath)['map']

    size1 = D.shape[0]
    size2 = D.shape[1]
    img_indice = np.zeros(shape=((size1)*(size2),2), dtype=np.int)
    gt = np.zeros(shape=((size1)*(size2),), dtype=np.int)
    t = 0

    for i in range(size1):
        for j in range(size2):
           img_indice[t] = np.array([i, j])    
           gt[t] = np.array(T[i, j])
           t += 1
            
    for label in range(3):
        indice_class = np.where(gt==label)[0]
        gt_class = gt[indice_class]
        imgind_class = img_indice[indice_class]
        np.save(savepath + 'imgind_class%s.npy'%label, imgind_class)
        np.save(savepath + 'gt_class%s.npy'%label, gt_class)
         
    print('=============end===============')
