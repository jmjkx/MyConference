from matplotlib import pyplot as plt
import scipy.io as sio

import numpy as np 
IMAGE = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell1-3.mat')['map']

img = np.zeros((973, 799, 3))
for i in range(973):
    for j in range(799):
        p = IMAGE[i, j]
        if p == 0:
            color = np.array([0, 0, 255])
        if p == 1:
            color = np.array([0, 255, 0])
        if p == 2:
            color = np.array([255, 0, 0])
        img[i, j ] = color
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('full.pdf') as pdf:
        fig = plt.figure()
        plt.imshow(img)
        height, width, channels = img.shape
        fig.set_size_inches(width / 100.0, height / 100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])  
        pdf.savefig(bbox_inches = 'tight')  # saves the current figure into a pdf page
        plt.close()
       

print('===========end=========')