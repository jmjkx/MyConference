import numpy as np
import pandas as pd

basepath = 'bloodcell2-2/Split/5/TrNumber_60/VaNumber_60/TeNumber_2000/paratune/'
score = []
t_path = ''
               
df = pd.DataFrame(data=np.random.randn(1, 1), 
                  index=['1'],
                  columns=['1']) 

for i in range(1, 33):
        t_path = basepath + 'mfa_Dim%s_kw4_kb26_PURE3DCNN_ConfMat.npy'%(str(i))
        conf_mat = np.load(t_path)
        score = conf_mat.trace() / conf_mat.sum()
        df.loc['1', str(i)] = score

df.to_csv('bloodcell2-2/Split/5/TrNumber_60/' \
                'VaNumber_60/TeNumber_2000/paratune/dim_result.csv')
print('=============result============')
# print('%.2f'%(np.mean(score)*100) + '±' +'%.2f'%(np.std(score, ddof=1)*100))
