import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from attention import CAM_Module, PAM_Module


class SAE_3DCNN_AM(nn.Module):
    def __init__(self):
        super().__init__()

        # spectral branch

        self.fc1 = nn.Linear(33, 20)
        # the second full connection, the first feature will be 20 that is the hidden nodes of the first hidden layer,  10 is determined as the size of second hidden layer, this will detect more features based on the first hidden layer
        self.fc2 = nn.Linear(20, 10)
        # start to decode or reconstruct the original input vector, the second feature will equal to the firs feature of fc2
        self.fc3 = nn.Linear(10, 20)
        # same reason, for the reconstruction of the input vector, the output vetor should have the same dimention as the input vector
        self.fc4 = nn.Linear(20, 60)
        # determin the activation function, sigmoid or you can use other activation function to compare
        self.activation = nn.Sigmoid()

        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 33), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        # 注意力机制模块

        # self.max_pooling2 = nn.MaxPool3d(kernel_size=(1, 1, 60))
        # self.avg_pooling2 = nn.AvgPool3d(kernel_size=(1, 1, 60))
        # self.max_pooling2 = nn.AdaptiveAvgPool3d(1)
        # self.avg_pooling2 = nn.AdaptiveAvgPool3d(1)

        self.conv25 = nn.Sequential(
                                nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                                          kernel_size=(3, 3, 2), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                # nn.Dropout(p=0.5),
                                nn.Linear(120, 3)# ,
                                # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(33)
        self.attention_spatial = PAM_Module(60)

        # fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, x):
        # spectral
        # 光谱注意力通道
        x1 = x[:, 0, 5, 5, :]
        x1 = self.activation(self.fc1(x1))
        x1 = self.activation(self.fc2(x1))
        x1 = self.activation(self.fc3(x1))
        # the final part of the decoding doesn't need to apply the activation function, we directly use full coneection fc4 function
        x1 = self.fc4(x1)
        for i in range(3):
            x1 = x1.unsqueeze(-1)
        x11 = self.attention_spectral(x1)
        x1 = torch.mul(x1, x11)
        # spatial
        # print('x', X.shape)
        x21 = self.conv21(x)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        # print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        # print('x25', x25.shape)
        # 空间注意力机制
        x2 = self.attention_spatial(x25)
        x2 = torch.mul(x2, x25)
        # model1
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)
        # print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        return output


class SAE_3DCNN_NEW(nn.Module):
    def __init__(self):
        super().__init__()

        # spectral branch

        self.fc1 = nn.Linear(33, 20)
        # the second full connection, the first feature will be 20 that is the hidden nodes of the first hidden layer,  10 is determined as the size of second hidden layer, this will detect more features based on the first hidden layer
        self.fc2 = nn.Linear(20, 10)
        # start to decode or reconstruct the original input vector, the second feature will equal to the firs feature of fc2
        self.fc3 = nn.Linear(10, 20)
        # same reason, for the reconstruction of the input vector, the output vetor should have the same dimention as the input vector
        self.fc4 = nn.Linear(20, 60)
        # determin the activation function, sigmoid or you can use other activation function to compare
        self.activation = nn.Sigmoid()

        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 33), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        # 注意力机制模块
        self.conv25 = nn.Sequential(
                                nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                                          kernel_size=(3, 3, 2), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                # nn.Dropout(p=0.5),
                                nn.Linear(120, 3)# ,
                                # nn.Softmax()
        )

        self.attention_spatial1 = MS_CAM(channels=60)
        self.attention_spatial2 = MS_CAM(channels=60)
        self.attention_spatial3 = MS_CAM(channels=60)
        self.attention_spectral = CAM_Module(33)
        # fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, x):
        # spectral
        # 光谱注意力通道
        x1 = x[:, 0, 5, 5, :]
        x1 = self.activation(self.fc1(x1))
        x1 = self.activation(self.fc2(x1))
        x1 = self.activation(self.fc3(x1))
        # the final part of the decoding doesn't need to apply the activation function, we directly use full coneection fc4 function
        x1 = self.fc4(x1)
        for i in range(3):
            x1 = x1.unsqueeze(-1)
       
        x21 = self.conv21(x)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)
       

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        # 空间注意力机制
        # x1 = self.global_pooling(x1)
        # x1 = x1.squeeze(-1)
        # x2 = self.global_pooling(x25)
        # x2 = x2.squeeze(-1)
        # x11 = self.attention_spectral(x1)
        # x1 = torch.mul(x1, x11)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)


        x2 = self.attention_spatial(x25)
     #   x2 = self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)
        # x_pre = self.global_pooling(x_pre)
        # x_pre = torch.flatten(x_pre, start_dim=1, end_dim=4)
        x_pre = torch.cat((x1, x2), dim=1)
        output = self.full_connection(x_pre)
        return output


class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei







class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=33, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


