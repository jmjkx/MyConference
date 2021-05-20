import numpy as np
import scipy.io as sio

from utils import plot

IMAGE = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell2-2.mat')['image']
GND = sio.loadmat('/home/liyuan/Programming/python/'\
                        '高光谱医学/高光谱医学数据/bloodcell1-3.mat')['map']

img = np.zeros(IMAGE.shape[0:2] + (3,), dtype=np.float32)
img[:,:,0] = IMAGE[:,:,10]/IMAGE[:,:,10].max()/5
img[:,:,1] = IMAGE[:,:,20]/IMAGE[:,:,20].max()/2
img[:,:,2] = IMAGE[:,:,30]/IMAGE[:,:,30].max()/5

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('./55.pdf') as pdf:
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
print('.pdf ' + 'has been saved') 


print('end')
