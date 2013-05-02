'''
Created on Apr 22, 2013

@author: gena
'''
'''
Created on Apr 21, 2013

@author: gena
'''
import cPickle,gzip
import numpy as np
import utils as ut
from autoencoder import *
from scipy.optimize import fmin_bfgs  ,fmin_l_bfgs_b


print "download data"
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()



ind1 = range(0,np.size(train_set[0],0))
ind2 = range(0,np.size(valid_set[0],0))
ind3 = range(0,np.size(test_set[0],0))

np.random.shuffle(ind1) 
np.random.shuffle(ind2) 
np.random.shuffle(ind3) 

X = train_set[0][ind1]
y = train_set[1][ind1]
 
X_v = valid_set[0][ind2]
y_v = valid_set[1][ind2]
 
X_t = test_set[0][ind3]
y_t = test_set[1][ind3]


nFeatures = 784
hidden_layer_size = 529
nClasses = 784

sparsity_param = 0.01
lambda_= 0.0001
beta = 3

batchSize = 100
num_iters = 20
alpha = 0.1

func = [sigmoid, sigmoidGradient]

scatterPlot2D(X, y)

w_ = np.array(np.random.rand((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses))/100

X_ = addOnes(X)
y_ = X

X_,y_ = createBatches(X_,y_,batchSize)
m = len(X_)

#momunetum
w_diff = np.zeros_like(w_)
w_old  = np.zeros_like(w_)
mu = 0.99 

for i in range(1,num_iters):
    #if i>1:
            #mu = 0.99 
    for j in range(0,m):
        print "iter: " + str(j) + " epoch: " + str(i)
        g = nnGrad(w_, X_[j],y_[j],lambda_,sparsity_param,beta,func,nFeatures,hidden_layer_size, nClasses)
        w_old = w_
        w_ = w_- alpha*g #+ mu*(w_diff)
        w_diff = w_ - w_old
     
        print nnCostFunction(w_, X_[j], y_[j], 0, 0.01, 0, func, nFeatures, hidden_layer_size,
 nClasses)
np.savetxt('test.txt', w_)
w1 = np.reshape(w_[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))    
visualizeNet(w1[1:,:])




