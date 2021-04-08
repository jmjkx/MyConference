from __future__ import print_function

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.testing._private.utils import KnownFailureException
from sklearn.utils.random import sample_without_replacement
from tensorboardX import SummaryWriter

from models import GetNetDim, Net
from mymodel import MyDataset


def validate(net, data_loader, set_name, classes_name):
    """
    :param net:
    :param data_loader:
    :param set_name:  eg: 'valid' 'train' 'tesst
    :param classes_name:
    :return:
    """
    net.eval()
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])

    for data in data_loader:
        images, labels = data
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        outputs.detach_()
        _, predicted = torch.max(outputs.data, 1)

        # 统计混淆矩阵
        for i in range(len(labels)):
            cate_i = labels[i]
            pre_i = predicted[i]
            conf_mat[cate_i, pre_i] += 1.0

    for i in range(cls_num):
        print('class:{:<10}, total num:{:<6}, correct num:{:5}  Recall: {:.2%} Precision: {:.2%}'.format(
            classes_name[i], np.sum(
                conf_mat[i, :]), conf_mat[i, i], conf_mat[i, i] / (1 + np.sum(conf_mat[i, :])),
            conf_mat[i, i] / (1 + np.sum(conf_mat[:, i]))))

    print('{} set Accuracy:{:.2%}'.format(
        set_name, np.trace(conf_mat) / np.sum(conf_mat)))

    return conf_mat, '{:.2}'.format(np.trace(conf_mat) / np.sum(conf_mat))


# 生成图像
def show_confMat(confusion_mat, classes, set_name, out_dir):

    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    cmap = plt.cm.get_cmap('Greys')
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(
                confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    plt.show()
    



if __name__ == '__main__':
    # from AutoGPU import autoGPU
    # autoGPU(1, 'auto')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default='bloodcell1-3', metavar='D',
                        help='d')
    parser.add_argument('--n', type=int,  default=200, metavar='N',
                        help='n')
    parser.add_argument('--r', type=int,  default=3, metavar='R',
                        help='r')
    parser.add_argument('--k', type=int,  default=4, metavar='K',
                        help='k')
    parser.add_argument('--i', type=int,  default=1, metavar='I',
                        help='i')
    parser.add_argument('--o', type=int,  default=3, metavar='O',
                        help='GPU')
    args = parser.parse_args()

    d = args.dataset
    r = args.r
    n = args.n
    k = args.k
    I = args.i
    O = args.o


    # 定义超参
    EPOCH = 600   #50   # 1000
    BATCH_SIZE = 1000
    classes_name = [str(c) for c in range(3)]  # 分类地物数量

    X_train = np.load('./%s/Split_n200/xtrain_n%s_r%s_k%s.npy' % (d,n,r,k))
    Y_train = np.load('./%s/Split_n200/ytrain_n%s_r%s_k%s.npy' % (d,n,r,k))
    X_test = np.load('./%s/Split_n200/xtest_n%s_r%s_k%s.npy' % (d,n,r,k))
    Y_test = np.load('./%s/Split_n200/ytest_n%s_r%s_k%s.npy' % (d,n,r,k))

    x_train=np.array(X_train)
    y_train=np.array(Y_train).astype('int16')
    x_test=np.array(X_test)
    y_test=np.array(Y_test).astype('int16')

    training_dataset = MyDataset(x_train, y_train)
    testing_dataset = MyDataset(x_test, y_test)
    # Data Loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=testing_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 检查cuda是否可用
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(O)
    use_cuda = torch.cuda.is_available()

    # 生成log
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_path = os.path.join(os.getcwd(), "log")
    log_dir = os.path.join(log_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)


    # ---------------------查看全连接层维数--------------------------
    cnn = GetNetDim()
    cnn.init_weights()  # 初始化权值
    cnn = cnn.double()
    cnn = cnn.cuda()
    cnn.train()

    sample_x = torch.from_numpy(x_train[0][np.newaxis,:,:,:])
    sample_x = sample_x.cuda()
    out = cnn(sample_x)
    # 自适应计算全连接层维数


    # ---------------------搭建网络--------------------------
    cnn = Net(out.shape[1])  # 创建CNN, 输入全连接层维数
    cnn.init_weights()  # 初始化权值
    cnn = cnn.double()




    # --------------------设置损失函数和优化器----------------------
    optimizer = optim.Adam(cnn.parameters(), lr = 0.0001)  # lr:(default: 1e-3)优化器
    criterion = nn.CrossEntropyLoss()  # 损失函数
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=EPOCH/2, gamma=0.5)  # 设置学习率下降策略

    # --------------------训练------------------------------
    # 使用GPU
    cnn = cnn.cuda()
    for epoch in range(EPOCH):
        loss_sigma = 0.0    # 记录一个epoch的loss之和
        correct = 0.0
        total = 0.0
        scheduler.step()  # 更新学习率

        for batch_idx, data in enumerate(train_loader):
            # 获取图片和标签
            inputs, labels = data
            
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()  # 清空梯度
            cnn = cnn.train()
            
            outputs = cnn(inputs)
            
            loss = criterion(outputs, labels.long())
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权值

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += ((predicted == labels).squeeze().sum()).item()
            loss_sigma += loss.item()

            # 每 BATCH_SIZE 个 iteration 打印一次训练信息，loss为 BATCH_SIZE 个 iteration 的平均   
        loss_avg = loss_sigma / BATCH_SIZE
        loss_sigma = 0.0
        print("Training: Epoch[{:03}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
            epoch + 1, EPOCH, batch_idx + 1, len(train_loader), loss_avg, correct / total))
        # 记录训练loss
        writer.add_scalars(
            'Loss_group', {'train_loss': loss_avg}, epoch)
        # 记录learning rate
        writer.add_scalar(
            'learning rate', scheduler.get_last_lr()[0], epoch)
        # 记录Accuracy
        writer.add_scalars('Accuracy_group', {
                        'train_acc': correct / total}, epoch)
        # 每个epoch，记录梯度，权值
        for name, layer in cnn.named_parameters():
            writer.add_histogram(
                name + '_grad', layer.grad.cpu().data.numpy(), epoch)
            writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

        # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
        if epoch % 1 == 0:
            loss_sigma = 0.0
            cls_num = len(classes_name)
            conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
            cnn.eval()
            for batch_idx, data in enumerate(test_loader):
                images, labels = data
                if(use_cuda):
                    images, labels = images.cuda(), labels.cuda()
                cnn = cnn.train()
                outputs = cnn(images)  # forward
            
                outputs.detach_()  # 不求梯度
                
                loss = criterion(outputs, labels.long())  # 计算loss
                loss_sigma += loss.item()

                _, predicted = torch.max(outputs.data, 1)  # 统计
                # labels = labels.data    # Variable --> tensor
                # 统计混淆矩阵
                for j in range(len(labels)):
                    
                    
                    cate_i = labels[j]
                    pre_i = predicted[j]
                    conf_mat[cate_i, pre_i] += 1.0
            print('{} set Accuracy:{:.2%}'.format(
                'Valid', conf_mat.trace() / conf_mat.sum()))
            # 记录Loss, accuracy
            writer.add_scalars(
                'Loss_group', {'valid_loss': loss_sigma / len(test_loader)}, epoch)
            writer.add_scalars('Accuracy_group', {
                            'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)
            
            with open('./%s/parameter/n200/k=%s_r=%s.txt'%(d,k,r),'w') as f:
                f.writelines(str(conf_mat.trace() / conf_mat.sum()))
            

    # ipdb.set_trace()
    # df_obj.loc[I,'acc'] = conf_mat.trace() / conf_mat.sum()
    # df_obj.loc[I,'n'] = n
    # df_obj.loc[I,'k'] = k
    # df_obj.loc[I,'r'] = r
    import subprocess

    import ipdb
    import pandas as pd
    import pyinotify
    from pandas import DataFrame

    # def onChange(env):
    #     print(env)
    #     df_obj = pd.read_excel('./result.xlsx')
    #     new=pd.DataFrame({'acc':conf_mat.trace() / conf_mat.sum(),
    #               'n':n,
    #               'k':k,
    #               'r':r}, index = [0]
    #                 ) 
    #     df_obj=df_obj.append(new,ignore_index=True)  
    #     df_obj.to_excel('result.xlsx', index=False, header=True)
    #     notifier.stop()
    # # if k == 2:
    #     onChange('start')
    # else:
    #     wm = pyinotify.WatchManager()


        # wm.add_watch('./result.xlsx', pyinotify.IN_CLOSE_WRITE, onChange)
        # notifier = pyinotify.Notifier(wm)
        # notifier.loop()
   

    # ----------------------- 保存模型 并且绘制混淆矩阵图 -------------------------
    # cnn_save_path = os.path.join(log_dir, 'net_params.pkl')
    # torch.save(cnn.state_dict(), cnn_save_path)

    # conf_mat_train, train_acc = validate(cnn, train_loader, 'train', classes_name)
    # #conf_mat_train, train_acc = F(cnn, train_loader, 'train', classes_name)
    # conf_mat_valid, valid_acc = validate(cnn, test_loader, 'test', classes_name)

    # show_confMat(conf_mat_train, classes_name, 'train', log_dir)

    # show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)
    print('Finished Training')
