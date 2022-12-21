import numpy as np


def onehot_bak(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()  # mask 中的1 放入0,2,4...(第0层)
    buf.ravel()[nmsk-1] = 1  # mask中的0, 作为1,放入另外一层
    return buf

    
def onehot(data, n=2):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + 1 - data.ravel()
    buf.ravel()[nmsk] = 1
    return buf
