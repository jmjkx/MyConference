import numpy as np
import scipy.io as sio

IMAGE = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell2-2.mat')['image']
GND = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell2-2.mat')['map']

C = ['XTeind_class', 'XTrind_class', 'XVaind_class',
         'YTe_class', 'YTr_class', 'YVa_class']

for i in range(1, 11):
    pp = 'bloodcell2-2/Split/%s/TrNumber_5/VaNumber_5/TeNumber_2000/'%str(i)
    for j in range(3):
        for c in C:
            d = np.load(pp + c + str(j) + '.npy')
            sio.savemat(pp + c + str(j) + '.mat', {'M': d})
print('end')
