import os
import re


def testGPU(id=0, mem_collect='auto'): 
    GPU_state = os.popen('nvidia-smi -i %s' % str(id)).read()
    GPU_type = re.findall('(?=TITAN\s).*?(?=\s+Off)', GPU_state)[0]
    Usage = re.findall('(\d+)(?=\s*MiB)', GPU_state)[0]
    Memory = re.findall('(\d+)(?=\s*MiB)', GPU_state)[1]
    IDLE =  int(Memory) - int(Usage) 
    if mem_collect == 'auto':
        if int(Usage) < 20:
            return 1, GPU_type, IDLE, int(Memory)
        else:
            return 0, GPU_type, IDLE, int(Memory)

    else:
        if IDLE > mem_collect:
            return 1, GPU_type, IDLE, int(Memory)
        else:
            return 0, GPU_type, IDLE, int(Memory)


def autoGPU(GPU_NUM=6, GPU_MEM='auto'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    ID = []
    for i in [8, 5, 4, 2, 1, 0, 3, 7 ,6]:
        is_val, GPU, Usa, Mem = testGPU(i, GPU_MEM)
        if is_val == 1:
            ID.append(str(i))
            print('已选择第{}张卡，型号为{}，{}MB/{}MB显存可用'.format(i, GPU, Usa, Mem))
            if len(ID) == GPU_NUM:
                break
    assert len(ID)==GPU_NUM, '你要求的显卡条件无法满足，找你老板买'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(ID)

    
if __name__ == '__main__':
    for id in range(9):
        print(testGPU(id))
