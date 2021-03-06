import numpy as np


def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()  # mask 中的1 放入0,2,4...(第0层)
    buf.ravel()[nmsk-1] = 1  # mask中的0, 作为1,放入另外一层
    return buf

    

