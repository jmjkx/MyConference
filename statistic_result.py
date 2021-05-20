import os
import pickle
import time

import numpy as np
from sklearn.metrics import accuracy_score

# from utils import DataResult

for ser in [30, 50, 70]:
    score = []
    exist = []
    for i in range(1, 11):
        tpath = './bloodcell1-3/Split/number/Tr_%s/Va_%s/Te2000/%s/result/PURE3DCNN/result.pkl'%(ser, ser, i)
        # tpath = './bloodcell1-3/Split/%s/' \ 
        #                     'TrNumber_%s/VaNumber_%s/TeNumber_2000/PURE3DCNN' \
        #                     '_ConfMat.npy'%(str(i), str(ser), str(ser))
        if os.path.exists(tpath):
            exist.append(tpath)
    if len(exist) == 10:
        for matpath in exist:
            with open(matpath, 'rb') as f:
                result = pickle.load(f)
            y_true = result.y_true
            y_pre = result.y_pre
            score.append(accuracy_score(y_true, y_pre))

        print('=============%s result============'%ser)
        print('%.2f'%(np.mean(score)*100) + '±' +'%.2f'%(np.std(score, ddof=1)*100))
    else:
        for matpath in exist:print(matpath)

# import os

# import numpy as np

# for ser in [1, 2, 3, 4, 5, 10, 20, 40, 60, 80]:
#     score = []
#     exist = []
#     for i in range(1, 11):
#         tpath = './bloodcell2-2/Split/%s/' \
#                             'TrNumber_%s/VaNumber_%s/TeNumber_2000/2DCNN' \
#                             '_ConfMat.npy'%(str(i), str(ser), str(ser))
#         # tpath = './bloodcell1-3/Split/%s/' \
#         #                     'TrNumber_%s/VaNumber_%s/TeNumber_2000/PURE3DCNN' \
#         #                     '_ConfMat.npy'%(str(i), str(ser), str(ser))
#         if os.path.exists(tpath):
#             exist.append(tpath)
#     if len(exist) == 10:
#         for matpath in exist:
#             conf_mat = np.load(matpath)
#             score.append(conf_mat.trace()/conf_mat.sum())

#         print('=============%s result============'%ser)
#         print('%.2f'%(np.mean(score)*100) + '±' +'%.2f'%(np.std(score, ddof=1)*100))
#     else:
#         for matpath in exist:print(matpath)
