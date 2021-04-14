import numpy as np
import scipy.io as sio

IMAGE = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell1-3.mat')['image']
GND = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell1-3.mat')['map']

C = ['XTeind_class', 'XTrind_class', 'XVaind_class',
         'YTe_class', 'YTr_class', 'YVa_class']

for i in range(1, 11):
    pp = 'bloodcell1-3/Split/%s/TrNumber_80/VaNumber_80/TeNumber_all/'%str(i)
    for j in range(3):
        for c in C:
            d = np.load(pp + c + str(j) + '.npy')
            sio.savemat(pp + c + str(j) + '.mat', {'M': d})
print('end')
