import numpy as np

QGAIN = np.loadtxt('normalized_gains/coarse_gains.txt')[3]
for b in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
    print(repr(b) + ' ' + repr(np.power(4.0,1.0-b)*QGAIN))
