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


f = gzip.open('/home/gena/lab/data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()



ind1 = range(0,np.size(train_set[0],0))
ind2 = range(0,np.size(valid_set[0],0))
#ind3 = range(0,np.size(test_set[0],0))

np.random.shuffle(ind1) 
np.random.shuffle(ind2) 
#np.random.shuffle(ind3) 

X = (train_set[0][ind1])
y = X#train_set[1][ind1]
 
X_v = valid_set[0][ind2]
y_v = X_v#valid_set[1][ind2]
 
#X_t = test_set[0][ind3]
#y_t = test_set[1][ind3]


nFeatures = 784
hidden_layer_size = 300
nClasses = 784

sparsity_param = 0.01
lambda_= 0
beta = 3

batchSize = 250
num_iters = 30
alpha = 0.01

func = [sigmoid, sigmoidGradient]

w_ = np.array(np.random.rand((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses))/100

#y_ = X
X = addOnes(X)
X_v = addOnes(X_v)

X,y = createBatches(X,y,batchSize)
X_v,y_v = createBatches(X_v,y_v,batchSize)

m = len(X)

def rmsprop(X,y,w):
        meanSqr = np.ones_like(w)
        for i in range(m):
            print "Iter: " + str(i)
            g = nnGrad(w,X[i],y[i],lambda_,sparsity_param,beta,func,nFeatures,hidden_layer_size,nClasses)
            meanSqr  = 0.9*meanSqr + 0.1*(np.power(g,2))
            w = w - alpha*(g/(np.sqrt(meanSqr) + 1e-8))
            #print "Batch cost: " + str(nnCostFunction(w_,X[i],y[i],lambda_,sparsity_param,beta,func,nFeatures,hidden_layer_size,nClasses))
        return w

error_validation = np.zeros(num_iters)
bestVerr = 0
vErrorInc = 0
maxNumberVerrorIncrease = 10
w_best = np.zeros_like(w_)


for i in range(0,num_iters):
    w_ = rmsprop(X,y,w_)


    error_validation[i] = pedictionError(y_v,\
             nnBatchPredict(X_v, w_,func[0],nFeatures,hidden_layer_size,nClasses,len(X_v),batchSize),len(X_v)) 
    if i > 0: 
        if  error_validation[i] < bestVerr: #error rate is lower
            bestVerr = error_validation[i]
            w_best = w_
            if vErrorInc !=0: #zeroise counter
                vErrorInc = 0
        else: #validation error is becoming worse
            vErrorInc +=1     
    else: #save layer and validation error on the first iterations
        bestVerr = error_validation[0]
        w_best = w_    
    print "validation error: " + str(error_validation[i])
    if vErrorInc ==  maxNumberVerrorIncrease:
            break    
    
#np.savetxt('test.txt', w_)
w1 = np.reshape(w_best[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))    
visualizeFilters(w1[1:,:])




