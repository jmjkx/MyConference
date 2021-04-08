import os

import numpy as np

score = []
exist = []
for i in range(1, 11):
    tpath = './bloodcell2-2/Split/%s/' \
                        'TrNumber_80/VaNumber_80/TeNumber_2000/2DCNN' \
                        '_ConfMat.npy'%str(i)
    if os.path.exists(tpath):
        exist.append(tpath)
if len(exist) == 10:
    for matpath in exist:
        conf_mat = np.load(matpath)
        score.append(conf_mat.trace()/conf_mat.sum())

    print('=============result============')
    print('%.2f'%(np.mean(score)*100) + '±' +'%.2f'%(np.std(score, ddof=1)*100))
else:
    for matpath in exist:print(matpath)
