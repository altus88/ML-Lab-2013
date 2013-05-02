'''
Created on Apr 23, 2013

@author: gena
'''
import numpy as np
import cPickle,gzip
from bhtsne import *
import matplotlib.pyplot as plt
import time

print "download data"
f = gzip.open('/home/gena/lab/data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


X = train_set[0]
y = train_set[1]
#calc_tsne(X[0:100])
nSamples = 1000
res = np.zeros((len(X[0:nSamples]),2))
y_ = y[0:nSamples]

#start_time = time.time()
tsne_result = enumerate(bh_tsne(X[0:nSamples],30,0))
#elsapsedTime = time.time() - start_time

#print str(elsapsedTime) 
for j,result in tsne_result:
    res[j] = result  
    
fig = plt.figure()
plt.scatter(res[:, 0], res[:, 1], c=y_)
fig.show()    
raw_input("Press ENTER to exit")
        