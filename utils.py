import multiprocessing as mp
import os
import pickle
from datetime import datetime

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset as BaseDataset


def normlize(patch):
    norm_patch = np.zeros(shape=patch.shape)
    for batchidx in range(patch.shape[0]):
        for l in range(patch.shape[1]):
            for w in range(patch.shape[2]):
                image = patch[batchidx, l, w, :]
                assert image.max()-image.min() != 0, 'aaaaa'
                image = (image-image.min())/(image.max()-image.min())
                norm_patch[batchidx, l, w, :] = image
    return norm_patch


def bd_subtask(image_classindlist, gt_class , IMAGE, patchsize, reduction_matrix):
    print('==================subtask number %s======================='%str(len(image_classindlist)))
    img_gt = zip(image_classindlist, gt_class)
    imgpatch= np.empty(shape=(len(image_classindlist), patchsize, patchsize, 33))
    map = []
    for idx, (img_indice, gt) in enumerate(img_gt):
        for i in range(patchsize):
            for j in range(patchsize):
                if any([img_indice[0]-patchsize//2 + i  < 0,
                        img_indice[0]-patchsize//2 + i > IMAGE.shape[0] - 1,
                        img_indice[1]-patchsize//2 + j < 0,
                        img_indice[1]-patchsize//2 + j > IMAGE.shape[1] - 1]):
                    imgpatch[idx, i, j, :] = IMAGE[img_indice[0], img_indice[1], :]
                else:
                    imgpatch[idx, i, j, :] = IMAGE[img_indice[0]-patchsize//2 + i, 
                                                   img_indice[1]-patchsize//2 + j, :]
        map.append(gt)    
    imgpatch = normlize(imgpatch)
    print('================subtask end=====================')
    return imgpatch, np.array(map)


def build_data(IMAGE, patchsize,  datapath, dataclass, reduction_matrix=None):
    sum = 0
    pklName = {'Tr': 'train_patchgt.pkl',
               'Te': 'test_patchgt.pkl',
               'Va': 'valid_patchgt.pkl',
               }
        
    if os.path.exists(datapath + pklName[dataclass]):
        with open(datapath + pklName[dataclass], 'rb') as f:
            pklfile = pickle.load(f)
        if reduction_matrix is not None:
            pklfile['patch'] = np.matmul(pklfile['patch'], reduction_matrix)
        return pklfile['patch'], pklfile['gt']
    else:
        for label in range(3):
            image_classindlist = list(np.load(datapath + 'X%sind_class%s.npy'%(dataclass, label)))
            gt_class = list(np.load(datapath  + '/Y%s_class%s.npy'%(dataclass, label)))
            sum += len(image_classindlist)
        imgpatch = np.empty(shape=(sum, patchsize, patchsize, 33))

        image_classindlist = []
        gt_class = []
        for label in range(3):
            image_classindlist += list(np.load(datapath + 'X%sind_class%s.npy'%(dataclass, label)))
            gt_class += list(np.load(datapath  + '/Y%s_class%s.npy'%(dataclass, label)))

        tasknumber = len(image_classindlist)//50000+1
        p = mp.Pool(tasknumber)
        print('starting')
        results = [p.apply_async(bd_subtask, args=(image_classindlist[i*50000: (i+1)*50000], 
                                                    gt_class[i*50000: (i+1)*50000], 
                                                    IMAGE, patchsize, reduction_matrix)) 
                for i in range(tasknumber)]
        p.close()
        p.join()
        results = [p.get() for p in results]
        for idx, (imp, imgt) in enumerate(results):
            imgpatch[idx*50000:idx*50000+imp.shape[0]] = imp
        pklfile = {'patch': imgpatch, 'gt': np.array(gt_class)} 
        with open(datapath + pklName[dataclass], 'wb') as f:
            pickle.dump(pklfile, f, pickle.HIGHEST_PROTOCOL)
        if reduction_matrix is not None:
            imgpatch = np.matmul(imgpatch, reduction_matrix)
        return imgpatch, np.array(gt_class)


class MyDataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(self, label_npy, *images_npy):
        self.images = []
        for idx, image in enumerate(images_npy):
            self.images.append(torch.from_numpy(image))
        self.labels = torch.from_numpy(label_npy)
        self.length = self.labels.shape[0]

    def __getitem__(self, i):
        image = []
        for idx, _ in enumerate(self.images):
            image.append(self.images[idx][i])
        label = self.labels[i]
        return image + [label]
        
    def __len__(self):
        return self.length


def plot(pos, y_pre, shape,savepath):
    img = np.zeros(shape[:2]+(3,))
    for p_idx, p in enumerate(y_pre):
        if p == 0:
            color = np.array([0, 0, 255])
        if p == 1:
            color = np.array([0, 255, 0])
        if p == 2:
            color = np.array([255, 0, 0])
        img[tuple(list(pos[p_idx]))] = color
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(savepath + '.pdf') as pdf:
        fig = plt.figure()
        plt.imshow(img)
        height, width, channels = img.shape
        fig.set_size_inches(width / 100.0, height / 100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])  
        pdf.savefig(bbox_inches = 'tight')  # saves the current figure into a pdf page
        plt.close()
    print(savepath + '.pdf ' + 'has been saved') 
