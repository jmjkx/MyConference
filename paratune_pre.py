import numpy as np
import pandas as pd

basepath = 'bloodcell2-2/Split/5/TrNumber_60/VaNumber_60/TeNumber_2000/paratune/'
score = []
t_path = ''
               
df = pd.DataFrame(data=np.random.randn(9, 8), 
                  index=['1','2','3','4','5','6','7','8','9'], 
                  columns=['2','4','6','8','10','12','14','16']) 

for i in range(1, 10):
    for j in range(2, 29, 2):
        t_path = basepath + 'mfa_Dim30_kw%s_kb%s_PURE3DCNN_ConfMat.npy'%(str(i), str(j))
        conf_mat = np.load(t_path)
        score = conf_mat.trace() / conf_mat.sum()
        df.loc[str(i), str(j)] = score

df.to_csv('bloodcell2-2/Split/5/TrNumber_60/' \
                'VaNumber_60/TeNumber_2000/paratune/result.csv')
print('=============result============')
# print('%.2f'%(np.mean(score)*100) + '±' +'%.2f'%(np.std(score, ddof=1)*100))
