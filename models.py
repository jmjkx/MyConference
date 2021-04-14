import math
from re import S

import numpy as np
import torch
import torch.nn as nn
import torchsummary
from sklearn import model_selection, svm
from sklearn.neighbors import KNeighborsClassifier
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init

from attention import AFF, DAF, MS_CAM, CAM_Module, PAM_Module, iAFF
from model_compoent import MS_CAM, SELayer


class CNN(nn.Module):
    def __init__(self):
        """构造函数，定义网络的结构"""
        super().__init__()
        self.conv1 = nn.Conv2d(33, 6, 1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        """前向传播函数"""
        x = F.max_pfool2d(F.relu(self.conv1(x)), (2, 2))
        # 第二次卷积+池化操作
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # 重新塑形,将多维数据重新塑造为二维数据，256*400
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # x.size()返回值为(256, 16, 5, 5)(16, 5, 5)，256是batch_size
        size = x.size()[1:]        # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 
    
class SCN(nn.Module):
    def __init__(self, sample): 
    # 定义全连接层维数接口
        super().__init__()
    # in_channels,out_channels,kernel_size,stride(default:1),padding(default:0)
        self.sample = sample
        self.conv1 = torch.nn.Sequential(
        SeparableConv2d(33, 64, 1, 1, 0),  # 1*1卷积核
        nn.ReLU(inplace=True),
        nn.GroupNorm(64, 64)
    ).double().to('cuda')

        self.conv2 = nn.Sequential(
            SeparableConv2d(33, 256, 1, 1, 0),
            nn.GroupNorm(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            SeparableConv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            SeparableConv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            SeparableConv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(64, 64)
        ).double().to('cuda')

        self.classifier = nn.Sequential(
            nn.Linear(self._get_sizes(), 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 3),
            nn.LogSoftmax(dim=1)
        ).double().to('cuda')

    def forward(self, x):
        x0 = xZero(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x)
        x12 = torch.cat((x1, x2), 1)
        xCat = x12.view(-1, self.numFeatures(x12))  # 特征映射一维展开
        output = self.classifier(xCat)
        return  output

    def numFeatures(self, x):
        size = x.size()[1:]  # 获取卷积图像的h,w,depth
        num = 1
        for s in size:
            num *= s
            # print(s)
        return num

    def _get_sizes(self):
        with torch.no_grad():
            x0 = xZero(self.sample).double().to('cuda')
            x1 = self.conv1(x0)
            x2 = self.conv2(self.sample)
            x12 = torch.cat((x1, x2), 1)
            xCat = x12.view(-1, self.numFeatures(x12))  # 特征映射一维展开
        return xCat.shape[1]

class SeparableConv2d(nn.Module):  # Depth wise separable conv
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        # 每个input channel被自己的filters卷积操作
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class DBDA_network(nn.Module):
    def __init__(self, band=33, classes=3):
        super(DBDA_network, self).__init__()

        # spectral branch

        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Conv3d(in_channels=48, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
                                    nn.BatchNorm3d(72, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Conv3d(in_channels=72, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm3d(96, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=96, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)) # kernel size随数据变化

        #注意力机制模块

        #self.max_pooling1 = nn.MaxPool3d(kernel_size=(7, 7, 1))
        #self.avg_pooling1 = nn.AvgPool3d(kernel_size=(7, 7, 1))
        self.max_pooling1 = nn.AdaptiveAvgPool3d(1)
        self.avg_pooling1 = nn.AdaptiveAvgPool3d(1)

        self.shared_mlp = nn.Sequential(
                                    nn.Conv3d(in_channels=60, out_channels=30,
                                            kernel_size=(1, 1, 1), stride=(1, 1, 1)),
                                    nn.Conv3d(in_channels=30, out_channels=60,
                                            kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )

        self.activation1 = nn.Sigmoid()


        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
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
                                nn.Linear(120, classes)# ,
                                # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

        #fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        # print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        # print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        # print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        # print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        # print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)
        x1 = torch.mul(x1, x16)
        # spatial
        # print('x', X.shape)
        x21 = self.conv21(X)
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

class SAE(nn.Module):
    # since we are doing an inheritance, the arguement need not be added, but need ,
    def __init__(self, dim):
        # use super() function to optimize the SAE
        super(SAE, self).__init__()
        # Linear class which will make the different full connections between layers
        #now define this architecture by choosing the number of layers and the hidden neurons in each of these hidden layers
        #fc is full connection, the first part of the neural network, which is between the input vector features that is the ratings of all the movies for one specific user
        #and the first hidden layer is shorter vetor than the input vetoc
        #the object below will represent the full connection between this first input vector features and the first encoded vector, call this full connection fc1   
        #the first feature is the input vector, the number of input features, second feature will be the number of nodes or neurons in the first hidden layer. This is the number of elements in the first encoded vector, the following number is not tuned, which is determined as experience
        #20 means 20 features or hidden nodes will ne chosen to make the first layer's process, it can be tuned
        self.fc1 = nn.Linear(dim, 128)
        #the second full connection, the first feature will be 20 that is the hidden nodes of the first hidden layer,  10 is determined as the size of second hidden layer, this will detect more features based on the first hidden layer
        self.fc2 = nn.Linear(128, 256)
        #start to decode or reconstruct the original input vector, the second feature will equal to the firs feature of fc2
        self.fc3 = nn.Linear(256, 128)
        #same reason, for the reconstruction of the input vector, the output vetor should have the same dimention as the input vector
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)
        #determin the activation function, sigmoid or you can use other activation function to compare
        self.activation = nn.Sigmoid()
    #The main purpose of this forward() function (forward propagation), it will return in the end the vector of predicted rating that we will compare them with real rating that is input vector
    #The second arguement x which is our input vector, which you will see that we will transform this input vector x by encoding it twice and decoding it twice to get the final ouptut vetor that is the decoded vector that was reconstructed 

    def forward(self, x):
        #need to modify or update x after each encoding and decoding
        x = x.flatten(1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        #start decoding it
        x = self.activation(self.fc3(x))
        #the final part of the decoding doesn't need to apply the activation function, we directly use full coneection fc4 function
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        #x is our vector of predicted ratings
        return x
        #The main purpose of this forward() function (forward propagation), it will return in the end the vector of predicted rating that we will compare them with real rating that is input vector
    #The second arguement x which is our input vector, which you will see that we will transform this input vector x by encoding it twice and decoding it twice to get the final ouptut vetor that is the decoded vector that was reconstructed 

    def predict(self, x): # x: visible nodes
        x = self.forward(x)
        return x

class _3DCNN_AM(nn.Module):
    def __init__(self):
        super().__init__()
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
                                nn.Linear(60, 3)# ,
                                # nn.Softmax()
        )

        self.attention1 = MS_CAM(channels=12)
        self.attention2 = MS_CAM(channels=12)
        self.attention3 = MS_CAM(channels=1)
        # fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, x):
        x21 = self.attention3(x)
        x21 = self.conv21(x)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)
        x22 = self.attention1(x22)
     

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)
        #x23 = self.attention2(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)
      

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        # 空间注意力机制
        # x1 = self.global_pooling(x1)
        # x1 = x1.squeeze(-1)
        # x2 = self.global_pooling(x25)
        # x2 = x2.squeeze(-1)
      
        x2 = self.global_pooling(x25)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        # x_pre = self.global_pooling(x_pre)
        # x_pre = torch.flatten(x_pre, start_dim=1, end_dim=4)
        output = self.full_connection(x2)
        return output




        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class _3DCNN(nn.Module):
    def __init__(self, dim):
        super(_3DCNN, self).__init__()
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(dim, 1, 1 ), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1))
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
                                nn.Linear(60, 3)# ,
                                # nn.Softmax()
        ) 

    def forward(self, x):
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
        x2 = self.global_pooling(x25)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)
        output = self.full_connection(x2)
        # 空间注意力机制
        return output


class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)
       

    def __init__(self, input_channels=33, n_classes=3, patch_size=11, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1)
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0)
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        #self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)
        self.attension1 = SELayer(channel=20)
        self.attension2 = MS_CAM(channels=35)
        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.attension1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # x = self.attension2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
      
        x = x.reshape(-1, self.features_size)
        #x = self.dropout(x)
        x = self.fc(x)
        return x


class SAE_AM(nn.Module):
    # since we are doing an inheritance, the arguement need not be added, but need ,
    def __init__(self, ):
        # use super() function to optimize the SAE
        super(SAE_AM, self).__init__()
        # Linear class which will make the different full connections between layers
        #now define this architecture by choosing the number of layers and the hidden neurons in each of these hidden layers
        #fc is full connection, the first part of the neural network, which is between the input vector features that is the ratings of all the movies for one specific user
        #and the first hidden layer is shorter vetor than the input vetoc
        #the object below will represent the full connection between this first input vector features and the first encoded vector, call this full connection fc1   
        #the first feature is the input vector, the number of input features, second feature will be the number of nodes or neurons in the first hidden layer. This is the number of elements in the first encoded vector, the following number is not tuned, which is determined as experience
        #20 means 20 features or hidden nodes will ne chosen to make the first layer's process, it can be tuned
        self.fc1 = nn.Linear(33, 20)
        #the second full connection, the first feature will be 20 that is the hidden nodes of the first hidden layer,  10 is determined as the size of second hidden layer, this will detect more features based on the first hidden layer
        self.fc2 = nn.Linear(20, 10)
        #start to decode or reconstruct the original input vector, the second feature will equal to the firs feature of fc2
        self.fc3 = nn.Linear(10, 20)
        #same reason, for the reconstruction of the input vector, the output vetor should have the same dimention as the input vector
        self.fc4 = nn.Linear(20, 3)
        #determin the activation function, sigmoid or you can use other activation function to compare
        self.activation = nn.Sigmoid()
    #The main purpose of this forward() function (forward propagation), it will return in the end the vector of predicted rating that we will compare them with real rating that is input vector
    #The second arguement x which is our input vector, which you will see that we will transform this input vector x by encoding it twice and decoding it twice to get the final ouptut vetor that is the decoded vector that was reconstructed 
        self.attention_spectral = CAM_Module(33)
        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                # nn.Dropout(p=0.5),
                                nn.Linear(120, 3)# ,
                                # nn.Softmax()
        )

    def forward(self, x):
        x1 = self.activation(self.fc1(x))
        x1 = self.activation(self.fc2(x1))
        x1 = self.activation(self.fc3(x1))
        # the final part of the decoding doesn't need to apply the activation function, we directly use full coneection fc4 function
        x1 = self.fc4(x1)
        for i in range(3):
            x1 = x1.unsqueeze(-1)
        x11 = self.attention_spectral(x1)
        x1 = torch.mul(x1, x11)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        
        return x1 


class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels=33, n_classes=3):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 32, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_1x1 = nn.Conv3d(
            1, 32, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(64, 128, (3, 3))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (3, 3))
        self.conv7 = nn.Conv2d(128, 128, (3, 3))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))
        self.fc = nn.Linear(75, 3)
        # self.lrn1 = nn.LocalResponseNorm(256)
        # self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)
        # Local Response Normalization
        x = F.relu(x)
        # First convolution
        x = self.conv1(x)
        # Local Response Normalization
        x = F.relu(x)
        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)
        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        x = x.reshape(-1, 75)
        x = self.fc(x)
        return x

class PURE3DCNN_2AM(nn.Module):
    def __init__(self, channel):
        super(PURE3DCNN_2AM, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(channel, 1, 1), stride=(1, 1, 1))
        # Dense block
        self.batch_norm1 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv3d(in_channels=24, out_channels=24, padding=(0, 1, 1),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv3d(in_channels=24, out_channels=3, padding=(1, 1, 0),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
                                    nn.BatchNorm3d(3, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Conv3d(in_channels=24, out_channels=24, padding=(1, 1, 1),
                                kernel_size=(3, 3, 3), stride=(1, 1, 1))

        self.conv5 = nn.Sequential(
                                nn.Conv3d(in_channels=24, out_channels=3, padding=(1, 1, 0),
                                          kernel_size=(2, 3, 3), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )
        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                # nn.Dropout(p=0.5),
                                nn.Linear(48, 3)# ,
                                # nn.Softmax()
        ) 
        self.attention = AFF(24)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        # x = self.conv2(x)
        x_res = self.conv2(x)
        x = self.attention(x, x_res)
        x = self.batch_norm2(x)
        
        # x = self.conv2(x)
        # x = self.batch_norm2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        # x_res = self.batch_norm3(x)
        # x = self.attention(x, x_res)


        x = self.global_pooling(x)
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        # output = self.full_connection(x)
        # 空间注意力机制
        return x 


class PURE3DCNN(nn.Module):
    def __init__(self, depth):
        super(PURE3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(depth, 1, 1), stride=(1, 1, 1))
        # Dense block
        self.batch_norm1 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv3d(in_channels=24, out_channels=24, padding=(0, 1, 1),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv3d(in_channels=24, out_channels=3, padding=(1, 1, 0),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
                                    nn.BatchNorm3d(3, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Conv3d(in_channels=24, out_channels=24, padding=(1, 1, 1),
                                kernel_size=(3, 3, 3), stride=(1, 1, 1))

        self.conv5 = nn.Sequential(
                                nn.Conv3d(in_channels=24, out_channels=3, padding=(1, 1, 0),
                                          kernel_size=(2, 3, 3), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )
        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                # nn.Dropout(p=0.5),
                                nn.Linear(48, 3)# ,
                                # nn.Softmax()
        ) 
        self.attention = AFF(24)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        # x_res = self.conv2(x)
        # x = self.attention(x, x_res)
        x = self.batch_norm2(x)
        
        # x = self.conv2(x)
        # x = self.batch_norm2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        # x_res = self.batch_norm3(x)
        # x = self.attention(x, x_res)


        x = self.global_pooling(x)
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        # output = self.full_connection(x)
        # 空间注意力机制
        return x 

class _1DCNN(nn.Module):
    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = x.squeeze(dim=-1).squeeze(dim=-1)
            # x = x.unsqueeze(1)
            x = self.conv1(x)
            x = F.relu(self.pool1(x))
            x = self.conv2(x)
            x = F.relu(self.pool2(x))
            x = self.conv3(x)
            x = F.relu(self.pool3(x))
        return x.numel() 

    def __init__(self, input_channels, n_classes=3, kernel_size=None, pool_size=None):
        super(_1DCNN, self).__init__()
        if kernel_size is None:
           # [In our experiments, k1 is better to be [ceil](n1/9)]
           kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
           # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
           # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
           pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv1 = nn.Conv1d(1, 64, kernel_size)
        self.pool1 = nn.MaxPool1d(pool_size)
        self.conv2 =  nn.Conv1d(64, 128, kernel_size)
        self.pool2 = nn.MaxPool1d(pool_size)
        self.conv3 =  nn.Conv1d(128, 64, kernel_size)
        self.pool3= nn.MaxPool1d(pool_size)
        self.drop = nn.Dropout(0.3)

        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)] x = x.squeeze(dim=-1).squeeze(dim=-1)
        # x = x.unsqueeze(1)
        x = x[:, :, :, 5, 5]
        x = x.squeeze(-1).squeeze(-1)
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.conv2(x)
        x = F.relu(self.pool2(x))
        x = self.conv3(x)
        x = F.relu(self.pool3(x))

        x = x.view(-1, self.features_size)
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

class _3DCNN_1DCNN(nn.Module):
    
    def __init__(self, input_channels, n_classes=3, kernel_size=None, pool_size=None):
        super(_3DCNN_1DCNN, self).__init__()
        if kernel_size is None:
           # [In our experiments, k1 is better to be [ceil](n1/9)]
           kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
           # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
           # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
           pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv11 = nn.Conv1d(1, 32, kernel_size)
        self.pool11 = nn.MaxPool1d(pool_size)
        self.conv12 =  nn.Conv1d(32, 64, kernel_size)
        self.pool12 = nn.MaxPool1d(pool_size)
        self.conv13 =  nn.Conv1d(64, 32, kernel_size)
        self.pool13= nn.MaxPool1d(pool_size)
        self.drop1 = nn.Dropout(0.3)

      
        # [n4 is set to be 100]
      


        self.conv21 = nn.Conv3d(in_channels=1, out_channels=32,
                                kernel_size=(input_channels, 1, 1), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm3d(32, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv3d(in_channels=32, out_channels=64, padding=(0, 1, 1),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm3d(64, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv3d(in_channels=64, out_channels=32, padding=(1, 1, 0),
                                kernel_size=(1, 3, 3), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm3d(32, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv3d(in_channels=24, out_channels=24, padding=(1, 1, 1),
                                kernel_size=(3, 3, 3), stride=(1, 1, 1))

        self.conv25 = nn.Sequential(
                                nn.Conv3d(in_channels=24, out_channels=3, padding=(1, 1, 0),
                                          kernel_size=(2, 3, 3), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )
        self.global_pooling = nn.AdaptiveAvgPool3d(1)

        self.features_size1 = self._get_flattened_size1()
        self.features_size2 = self._get_flattened_size2()
        self.fc1 = nn.Linear(self.features_size1 + self.features_size2, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x_3d, x_1d):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)] x = x.squeeze(dim=-1).squeeze(dim=-1)
        # x = x.unsqueeze(1)
        x_1d = self.conv11(x_1d)
        x_1d = F.relu(self.pool11(x_1d))
        x_1d = self.conv12(x_1d)
        x_1d = F.relu(self.pool12(x_1d))
        x_1d = self.conv13(x_1d)
        x_1d = F.relu(self.pool13(x_1d))
        x_1d = x_1d.view(-1, self.features_size1)
        # x_1d = torch.tanh(self.fc11(x_1d))
        x_3d = self.conv21(x_3d)
        x_3d = self.batch_norm21(x_3d)
        x_3d = self.conv22(x_3d)
        x_3d = self.batch_norm22(x_3d)
        x_3d = self.conv23(x_3d)
        x_3d = self.batch_norm23(x_3d)
        x_3d = x_3d.view(-1, self.features_size2)
        # x_3d = self.global_pooling(x_3d)
        # x_3d = x_3d.squeeze(-1).squeeze(-1).squeeze(-1)
        x = torch.cat([0.3*x_1d, x_3d], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        # output = self.full_connection(x)
        # 空间注意力机制
        return x

    def _get_flattened_size1(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = x.squeeze(dim=-1).squeeze(dim=-1)
            # x = x.unsqueeze(1)
            x = self.conv11(x)
            x = F.relu(self.pool11(x))
            x = self.conv12(x)
            x = F.relu(self.pool12(x))
            x = self.conv13(x)
            x = F.relu(self.pool13(x))
        return x.numel() 

    def _get_flattened_size2(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels, 11, 11)
            x = self.conv21(x)
            x = self.batch_norm21(x)
            x  = self.conv22(x)
            x  = self.batch_norm22(x)
            x  = self.conv23(x)
            x  = self.batch_norm23(x)
        return x.numel() 

    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)


class _2DCNN(nn.Module):
    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, 11, 11)
            # x = x.squeeze(dim=-1).squeeze(dim=-1)
            # x = x.unsqueeze(1)
            x = self.conv1(x)
            x = F.relu(self.pool1(x))
            x = self.conv2(x)
            x = F.relu(self.pool2(x))
            x = self.conv3(x)
            x = F.relu(self.pool3(x))
        return x.numel() 

    def __init__(self, input_channels, n_classes=3, kernel_size=None, pool_size=None):
        super(_2DCNN, self).__init__()
        if kernel_size is None:
           # [In our experiments, k1 is better to be [ceil](n1/9)]
           kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
           # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
           # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
           pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size)
        self.pool1 = nn.MaxPool2d(pool_size)
        self.conv2 =  nn.Conv2d(64, 128, kernel_size)
        self.pool2 = nn.MaxPool2d(pool_size)
        self.conv3 =  nn.Conv2d(128, 64, kernel_size)
        self.pool3= nn.MaxPool2d(pool_size)
        self.drop = nn.Dropout(0.3)

        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)] x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.squeeze(1)
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.conv2(x)
        x = F.relu(self.pool2(x))
        x = self.conv3(x)
        x = F.relu(self.pool3(x))
        x = x.view(-1, self.features_size)
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

def mysvm(x_train, y_train, x_test):
    SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
                                       'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                   {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]
    class_weight = 'balanced'
   
    try:
        clf = svm.SVC(class_weight=class_weight)
        clf = model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4)
        clf.fit(x_train, y_train)
    except ValueError:
        clf = svm.SVC(class_weight=class_weight)
        clf.fit(x_train, y_train)
    # print("SVM best parameters : {}".format(clf.best_params_))
    prediction = clf.predict(x_test)
    return prediction

def myknn(x_train, y_train, x_test):
    kclf = KNeighborsClassifier(n_neighbors=1)
    kclf.fit(x_train, y_train)
    prediction = kclf.predict(x_test)   
    return prediction 
