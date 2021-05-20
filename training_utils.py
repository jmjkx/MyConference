import glob
import os
import random
from datetime import datetime

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.nn import init
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm

from utils import DataResult, MyDataset


def setup_seed(seed):
   np.random.seed(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.cuda.manual_seed_all(seed) 
   torch.backends.cudnn.benchmark = True 
#    torch.backends.cudnn.deterministic = True    
#    torch.backends.cuda.matmul.allow_tf32 = True
#    torch.backends.cudnn.allow_tf32 = True



        
class TrainProcess():
    def __init__(self, model, mixdata, train_config, writerpath) -> None:
        super().__init__()
        self.data_mix = mixdata
        self.train_config = train_config
        self.model = model
        self.writerpath = writerpath
        self.criterion = nn.CrossEntropyLoss()
        self.train_result = DataResult()
        self.valid_result = DataResult()
        self.test_result = DataResult()
        pass

    def training_start(self):
        print('--------------------------训练----------------------------')
        # 使用GPU
        with open(self.train_config) as file:
            dict = file.read()
            config = yaml.load(dict, Loader=yaml.FullLoader)
        learning_rate = config['learning_rate']
        EPOCH = config['epoch']    
        TRAIN_BATCHSIZE = config['train_batchsize'] 
        TEST_BATCHSIZE = config['test_batchsize']
        VALID_BATCHSIZE = config['valid_batchsize']
        writer_path = self.writerpath
        OPTIMIZER = config['optimization'] 
        writer = SummaryWriter(writer_path + 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S"))  
        # writer.add_graph(self.model, torch.rand((1,) + self.data_mix['train_patch'].shape[1:]).cuda()) 
        if OPTIMIZER == 'Adam':
            # optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  
            optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0)
        elif OPTIMIZER == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate) 
        #  损失函数
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, 15, eta_min=0.0, last_epoch=-1)

        training_dataset = MyDataset(self.data_mix['train_gt'], self.data_mix['train_patch'])
        self.train_loader = torch.utils.data.DataLoader(
            dataset=training_dataset,
            batch_size=TRAIN_BATCHSIZE,
            shuffle=True
        )
        test_dataset = MyDataset(self.data_mix['test_gt'], self.data_mix['test_patch'])
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=TEST_BATCHSIZE,
            shuffle=False
        )
        
        if self.data_mix['valid_gt'].shape[0] != 0:
            valid_dataset = MyDataset(self.data_mix['valid_gt'], self.data_mix['valid_patch'])
            self.valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=VALID_BATCHSIZE,
                shuffle=True
            )
        else:
            valid_dataset = None

        self.model = self.model.to('cuda')
        self.model = self.model.double() 
        best_validacc = 0
       
        for epoch in range(EPOCH):
            trainloss_sigma = 0.0    # 记录一个epoch的loss之和
            for batch_idx, data in enumerate(self.train_loader):
                for idx, item in enumerate(data):
                    data[idx] = item.to('cuda')
                *inputs, labels = data
                optimizer.zero_grad()
                # 清空梯度
                self.model.train()
                outputs = self.model(*inputs)
                # outputs.retain_grad()
                loss = self.criterion(outputs, labels.long()) 
                loss.backward(retain_graph=True)  # 反向传播 optimizer.step()  # 更新权值
                optimizer.step()  
                # 统计预测信息
                trainloss_sigma += loss.item()
                # 每 BATCH_SIZE 个 iteration 打印一次训练信息，loss为 BATCH_SIZE 个 iteration 的平均   
            loss_avg = trainloss_sigma / TRAIN_BATCHSIZE
            train_acc = self.evaluate(self.train_loader, self.train_result)
            writer.add_scalar('Train Acc', train_acc, global_step=epoch+1) 
            writer.add_scalar('Train Loss', loss_avg, global_step=epoch+1) 
            print("Training: Epoch[{:03}/{:0>3}] Loss: {:.8f} Acc:{:.2%} Lr:{:.2}".format(
            epoch + 1, EPOCH,  loss_avg, train_acc, optimizer.state_dict()['param_groups'][0]['lr']))
            scheduler.step(loss_avg)  # 更新学习率
        # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
            if valid_dataset:
                valid_acc = self.evaluate(self.valid_loader, self.valid_result)
                writer.add_scalar('Valid Acc', valid_acc, global_step=epoch+1)
                print('{} set Accuracy:{:.2%}'.format('Valid', valid_acc))
                if valid_acc > best_validacc:
                    print("Higher Valid Accuracy:{:.2%}, Old Valid Accuracy:{:.2%}".format(valid_acc, best_validacc))
                    best_validacc = valid_acc
                    self.bestmodel = self.model.state_dict()
        print('===================Training Finished ======================')
        if valid_dataset:
            self.model.load_state_dict(self.bestmodel)
        test_acc = self.evaluate(self.test_loader, self.test_result) 
        if valid_dataset:
            print('Best {} set Accuracy:{:.2%}'.format('Valid', best_validacc)) 
        print('{} set Accuracy:{:.2%}'.format('Test', test_acc))
        writer.close()

    def evaluate(self, test_loader, data_result: DataResult):
        '''
        返回accf
        '''
        data_result.refresh()
        loss_sigma = 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                for idx, item in enumerate(data):
                    data[idx] = item.to('cuda')
                *images, labels = data
                self.model.eval()
                outputs = self.model(*images)  # forward
                #outputs.detach_()  # 不求梯度
                loss = self.criterion(outputs, labels.long())  # 计算loss
                loss_sigma += loss.item()
                _, predicted = torch.max(outputs.data, 1)  # 统计
                # 统计混淆矩阵
                data_result.y_pre+= list(predicted.cpu().numpy())
                data_result.y_true += list(labels.cpu().numpy())
        data_result.get_confmat()
        return data_result.conf_mat.trace() / data_result.conf_mat.sum()

    @staticmethod      
    def __init__weight(model):
        for name, param in model.named_parameters():
            init.normal_(param, mean=0, std=0.01)
            print(name, param.data)

 
if __name__ == '__main__':
    a = DataResult()
    print('end')
