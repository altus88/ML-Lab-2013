'''
Created on Apr 21, 2013

@author: gena
'''
import cPickle,gzip
import numpy as np
import utils as ut
from autoencoder import *


print "download data"
f = gzip.open('cifar-10-python.tar.gz', 'rb')
cPickle.load(f)
f.close()



# ind1 = range(0,np.size(train_set[0],0))
# ind2 = range(0,np.size(valid_set[0],0))
# ind3 = range(0,np.size(test_set[0],0))
# 
# np.random.shuffle(ind1) 
# np.random.shuffle(ind2) 
# np.random.shuffle(ind3) 
# 
# X = train_set[0][ind1]
# y = train_set[1][ind1]
#  
# X_v = valid_set[0][ind2]
# y_v = valid_set[1][ind2]
#  
# X_t = test_set[0][ind3]
# y_t = test_set[1][ind3]


#ut.scatterPlot2D(X[0:1000], y[0:1000])
#v = nnGrad(W,X,y,lambda_,f,f_grad,nFeatures,hidden_layer_size,nClasses):

