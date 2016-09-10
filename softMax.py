import numpy as np
def softmax(x):
    if x.ndim <= 1:
        x = x - np.max(x)
        ex = np.exp(x)
        dist =  ex / np.sum(ex)
    else:
        maxes = np.amax(x, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        e = np.exp(x - maxes)
        dist = e / np.sum(e, axis=1)
    return dist
#import pdb
#pdb.set_trace()
print softmax(np.array([[101,102],[-1,-2]]))




