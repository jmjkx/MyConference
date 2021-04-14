import glob
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
from tqdm import tqdm


def get_posgt(d, dataclass):
    pos_files = sorted(glob.glob(d +'X%sind_class*.npy'%dataclass), key=lambda x: int(x[-5]))
    datasetPos = []
    for f in pos_files:
        datasetPos += list(np.load(f))
    gt_files = sorted(glob.glob(d +'Y%s_class*.npy'%dataclass), key=lambda x: int(x[-5]))
    datasetGt = []
    for f in gt_files:
        datasetGt += list(np.load(f))
    return datasetPos, datasetGt


def normlize(patch):
    norm_patch = np.zeros(shape=patch.shape)
    for batchidx in range(patch.shape[0]):
        if len(patch[0].shape) > 2:
            norm_patch[batchidx] = normlize(patch[batchidx])
        else:
            image = patch[batchidx, :]
            image = (image-image.min())/(image.max()-image.min())
            norm_patch[batchidx, :] = image
    return norm_patch


def bd_subtask(image_classindlist, gt_class , IMAGE, patchsize):
    img_gt = zip(image_classindlist, gt_class)
    imgpatch= np.empty(shape=(len(image_classindlist), patchsize, patchsize, IMAGE.shape[2]))
    for idx, (img_indice, gt) in enumerate(tqdm(img_gt)):
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
    imgpatch = normlize(imgpatch)
    return imgpatch


class DataPreProcess(object):
    def __init__(self,
                 IMAGE,
                 patchsize, 
                 datapath,
                 dataclass,
                 reduction_matrix=None,
                 tasknum=20) -> None:
        self.IMAGE = IMAGE
        self.patchsize = patchsize
        self.datapath = datapath
        self.dataclass = dataclass
        self.reduction_matrix = reduction_matrix
        self.tasknum = tasknum
        self._build()
        pass
    
    def _build(self):
        pklName = {'Tr': 'TrainData.pkl',
                   'Te': 'TestData.pkl',
                   'Va': 'ValidData.pkl', }
        
        try:
            with open(self.datapath + pklName[self.dataclass], 'rb') as f:
                pklfile = pickle.load(f)
            if self.reduction_matrix is not None:
                pklfile['patch'] = np.matmul(pklfile['patch'], self.reduction_matrix)
            self.patch = pklfile['patch']
            self.gt = pklfile['gt']
            self.pos = pklfile['pos']
            print('Lucky Dog! ' + self.dataclass + ' data already exists!')

        except (FileNotFoundError, KeyError):
            self.pos, self.gt = get_posgt(self.datapath, self.dataclass)
            sample_number = len(self.pos)
            imgpatch = np.empty(shape=(sample_number, self.patchsize, self.patchsize, self.IMAGE.shape[2]))
            interval = sample_number//self.tasknum
            interval += 1
            print('=================== {0} {2} samples to process with {1} multi-process  ===================='.format(len(self.pos), self.tasknum, self.dataclass))
            p = mp.Pool(self.tasknum)
            print('starting')
            results = [p.apply_async(bd_subtask, args=(self.pos[i*interval: (i+1)*interval],
                                                       self.gt[i*interval: (i+1)*interval],
                                                       self.IMAGE, self.patchsize))
                       for i in range(self.tasknum)]
            p.close()
            p.join()
            results = [p.get() for p in results]
            for idx, imp in enumerate(results):
                imgpatch[idx*interval:idx*interval+imp.shape[0]] = imp
            pklfile = {'patch': imgpatch, 'gt': np.array(self.gt), 'pos': np.array(self.pos)} 
            with open(self.datapath + pklName[self.dataclass], 'wb') as f:
                pickle.dump(pklfile, f, pickle.HIGHEST_PROTOCOL)
            if self.reduction_matrix is not None:
                imgpatch = np.matmul(imgpatch, self.reduction_matrix)
            self.patch = pklfile['patch']
            self.gt = pklfile['gt']
            self.pos = pklfile['pos']
            print(self.dataclass + ' data has been built!')

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


def plot(pos: list, y_pre: list, shape,savepath):
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
