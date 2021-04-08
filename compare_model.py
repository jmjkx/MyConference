import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from attention import CAM_Module, PAM_Module


class SAE_3DCNN(nn.Module):
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

        # model1
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.global_pooling(x25)
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
