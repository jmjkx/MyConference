import glob
import multiprocessing as mp
import os
import pickle
import time
from datetime import datetime

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm


class DataResult():
    conf_mat = np.zeros([3, 3])
    y_pre = []
    y_true = []

    def refresh(self):
        self.conf_mat = np.zeros([3, 3])
        self.y_pre = []
        self.y_true = []
 
    def get_confmat(self):
        self.conf_mat = confusion_matrix(self.y_true, self.y_pre,)


class ProcessedData(object):
    def __init__(self, patch, gt, pos) -> None:
        super().__init__()
        self.patch = patch
        self.gt = gt
        self.pos = pos

def normlize(patch):
    norm_patch = np.zeros(shape=patch.shape)
    for batchidx in range(patch.shape[0]):
        if len(patch.shape) > 2:
            norm_patch[batchidx] = normlize(patch[batchidx])
        else:
            image = patch[batchidx, :]
            image = (image-image.min())/(image.max()-image.min())
            norm_patch[batchidx, :] = image
    return norm_patch


def bd_subtask(image_classindlist, IMAGE, patchsize):
    imgpatch= np.empty(shape=(len(image_classindlist), patchsize, patchsize, IMAGE.shape[2]))
    for idx, img_indice in enumerate(tqdm(image_classindlist)):
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
                 IMAGE: np.array,
                 patchsize: int,
                 datapath: str,
                 tasknum=20) -> None:
        self.IMAGE = IMAGE
        self.patchsize = patchsize
        self.datapath = datapath
        self.tasknum = tasknum
        self._build()
        pass
    
    def _build(self):
        try:
            with open(self.datapath + 'PatchGt_%s.pkl'%str(self.patchsize), 'rb') as f:
                pklfile = pickle.load(f)
            print('Lucky Dog! Patch data already exists!')
            self.processeddata = pklfile

        except (FileNotFoundError, KeyError):
            with open(self.datapath + 'spliteddata.pkl', 'rb') as f:
                splitimggt = pickle.load(f)
            trainpos, traingt = self.parsespdata(splitimggt['train'])
            testpos, testgt = self.parsespdata(splitimggt['test'])
            traingt = np.array(traingt)
            testgt = np.array(testgt)
            if splitimggt['valid'] is None:
                validpos = np.empty((0,))
                validgt = np.empty((0,))
            else:
                validpos, validgt = self.parsespdata(splitimggt['valid'])
                validgt = np.array(validgt)
            
            # def mysort(x):
            #     score = x[0][0] + x[0][1]/10000
            #     return score
            
            # def newsequence(pklfile):
            #     pos = []
            #     gt = []
            #     for key, value in pklfile.items():
            #             pos += list(value[0])
            #             gt += list(value[1])
            #     mixdata = zip(pos, gt)
            #     mixdata_list = [(pos, gt) for pos, gt in mixdata]
            #     mixdata_list = sorted(mixdata_list, key=mysort)
            #     pos_list = [x[0] for x in mixdata_list]
            #     gt_list = [x[1] for x in mixdata_list]
            #     return np.array(pos_list), np.array(gt_list)

            # trainpos, traingt = newsequence(splitimggt['train'])
            # testpos, testgt = newsequence(splitimggt['test'])

            trainpatch = self.getpatch(trainpos, self.IMAGE, 'Training')
            validpatch = self. getpatch(validpos, self.IMAGE, 'Valid')
            testpatch = self.getpatch(testpos, self.IMAGE, 'Test')
            self.processeddata = {}
            self.processeddata['train'] = ProcessedData(trainpatch, traingt, trainpos)
            self.processeddata['test'] = ProcessedData(testpatch, testgt, testpos)
            self.processeddata['valid'] = ProcessedData(validpatch, validgt, validpos)
            # with open(self.datapath + 'PatchGt_%s.pkl'%str(self.patchsize), 'wb') as f:
            #     pickle.dump(self.processeddata, f, pickle.HIGHEST_PROTOCOL)
            print('Patch data has been built!')

    def parsespdata(self, spdata) -> list:
        pos = []
        gt = []
        for _, value in spdata.items():
            pos = pos + list(value[0])
            gt = gt + list(value[1])
        return pos, gt

    def getpatch(self, posdata, image, dataname):
        if posdata is None:
            return None
        else:
            sample_number = len(posdata)
            imgpatch = np.empty(shape=(sample_number, self.patchsize, self.patchsize, self.IMAGE.shape[2]))
            interval = sample_number//self.tasknum
            interval += 1
            print('=================== {0} {2} samples to process with {1} multi-process  ===================='.format(len(posdata), self.tasknum, dataname))
            p = mp.Pool(self.tasknum)
            print('starting')
            results = [p.apply_async(self.bd_subtask, args=(posdata[i*interval: (i+1)*interval],
                                                        image, self.patchsize))
                        for i in range(self.tasknum)]
            p.close()
            p.join()
            results = [p.get() for p in results]
            for idx, imp in enumerate(results):
                    imgpatch[idx*interval:idx*interval+imp.shape[0]] = imp
            return imgpatch

    def bd_subtask(self, image_classindlist, IMAGE, patchsize):
        imgpatch= np.empty(shape=(len(image_classindlist), patchsize, patchsize, IMAGE.shape[2]))
        for idx, img_pos in enumerate(tqdm(image_classindlist)):
            for i in range(patchsize):
                for j in range(patchsize):
                    if any([img_pos[0]-patchsize//2 + i  < 0,
                            img_pos[0]-patchsize//2 + i > IMAGE.shape[0] - 1,
                            img_pos[1]-patchsize//2 + j < 0,
                            img_pos[1]-patchsize//2 + j > IMAGE.shape[1] - 1]):
                        imgpatch[idx, i, j, :] = IMAGE[img_pos[0], img_pos[1], :]
                    else:
                        imgpatch[idx, i, j, :] = IMAGE[img_pos[0]-patchsize//2 + i, 
                                                        img_pos[1]-patchsize//2 + j, :]
        imgpatch = normlize(imgpatch)
        return imgpatch


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


def plot(pos: list, y_pre: list, shape, savepath):
    img = np.zeros(shape[:2]+(3,), dtype=np.int)
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


def splitdata(image,
              groudtruth,
              savepath,
              trainnum=0.1,
              validnum=0.1,
              testnum=0.8,
              ):
    from sklearn.model_selection import train_test_split
    from sklearn.utils.random import sample_without_replacement

    if os.path.exists(savepath + 'spliteddata.pkl'):
        print('恭喜你， 划分数据(未取patch)已经存在')
        with open(savepath + 'spliteddata.pkl', 'rb') as f:
            return pickle.load(f)
            
    size1 = image.shape[0]
    size2 = image.shape[1]
    img_pos = np.zeros(shape=((size1)*(size2),2), dtype=np.int)
    gt = np.zeros(shape=((size1)*(size2),), dtype=np.int)
    t = 0
    for i in range(size1):
        for j in range(size2):
           img_pos[t] = np.array([i, j])    
           gt[t] = np.array(groudtruth[i, j])
           t += 1

    spliteddata = {'train':{}, 'valid':{}, 'test':{}}
    for label in range(0, groudtruth.max()+1):
        indice_class = np.where(gt==label)[0]
        gt_class = gt[indice_class]
        imgpos_class = img_pos[indice_class]
        samplepos = None

        nte = imgpos_class.shape[0] - trainnum - validnum if testnum == -1 else testnum

        if trainnum+validnum+nte> 1 :
            samplepos = sample_without_replacement(imgpos_class.shape[0],
                                                    trainnum+validnum+nte)
        elif trainnum+validnum+nte< 1:
            samplepos = sample_without_replacement(imgpos_class.shape[0],
                                             imgpos_class.shape[0]*(trainnum+validnum+nte))
        if samplepos is not None:
            imgpos_class = imgpos_class[samplepos]
            gt_class = gt_class[samplepos]


        pos_train, pos_teva,\
        y_train, y_teva = train_test_split(imgpos_class, gt_class, 
                                           train_size = trainnum,
                                           random_state = time.localtime(time.time()).tm_sec, 
                                           stratify = gt_class)
        spliteddata['train'].update({str(label):(pos_train, y_train)})

        
        testsize = nte / (validnum + nte) if nte < 1 else nte

        if testsize == 1:
            pos_test, y_test = pos_teva, y_teva
            spliteddata['valid'] = None
        else:
            pos_valid, pos_test,\
            y_valid, y_test = train_test_split(pos_teva, y_teva, 
                                                    test_size = testsize,
                                                    random_state = time.localtime(time.time()).tm_sec, 
                                                    stratify = y_teva) 
            spliteddata['valid'].update({str(label):(pos_valid, y_valid)})
        spliteddata['test'].update({str(label):(pos_test, y_test)})
        
    with open(savepath + 'spliteddata.pkl', 'wb') as f:
            pickle.dump(spliteddata, f, pickle.HIGHEST_PROTOCOL)
    print('=============split finished===============')
    return spliteddata

def setpath(dataset, trialnumber, NTr, NVa, NTe, modelname):
    foldertype = 'proportion' if NTe + NTr + NVa <= 1 else 'number'
    if NTe == -1:
        datapath = './' + dataset + '/Split/'+ foldertype + '/Tr_%s/' % NTr + 'Va_%s/' % NVa + 'Te_all/%s/'%str(trialnumber)
    else:
        datapath = './' + dataset + '/Split/' + foldertype + '/Tr_%s/' % NTr + 'Va_%s/' % NVa + 'Te%s/%s/' % (NTe, str(trialnumber))
    resultpath = datapath + 'result/%s/'%modelname
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    imagepath = datapath + 'image/%s/'%modelname 
    if not os.path.exists(imagepath):
        os.makedirs(imagepath)
    return resultpath, imagepath, datapath

def myplot(processeddata, IMAGE, imagepath, trainingresult: DataResult):
    imgPos = np.array(list(processeddata['train'].pos) + list(processeddata['valid'].pos) + list(processeddata['test'].pos))
    imgGt = np.array(list(processeddata['train'].gt) + list(processeddata['valid'].gt) + trainingresult.y_pre)
    plot(imgPos, imgGt, IMAGE.shape, imagepath + 'testprediction')
    imgPos = np.array(list(processeddata['train'].pos) + list(processeddata['valid'].pos))
    imgGt = np.array(list(processeddata['train'].gt) + list(processeddata['valid'].gt))
    plot(imgPos, imgGt, IMAGE.shape, imagepath + 'traindata')
    imgPos = np.array(list(processeddata['train'].pos) + list(processeddata['valid'].pos) + list(processeddata['test'].pos))
    imgGt = np.array(list(processeddata['train'].gt) + list(processeddata['valid'].gt) + trainingresult.y_true)
    plot(imgPos, imgGt, IMAGE.shape, imagepath + 'groundtruth')



if __name__ == '__main__':
    IMAGE = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell1-3.mat')['image']
    GND = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell1-3.mat')['map']
    splitdata(IMAGE,
              GND,
              './test/',
              trainnum=0.2,
              validnum=0,
              testnum=0.8
              )

    D = DataPreProcess(IMAGE, datapath='./test/', patchsize=7)
