'''
Created on Apr 15, 2013

@author: gena
'''
from utils import addOnes,mapClasses,miniBatchGradientDescent
import cPickle,gzip

def train(alpha,num_iters,batchSize,lambda_,method,path):
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
     
    X = train_set[0]
    y = train_set[1]
     
    X_v = valid_set[0]
    y_v = valid_set[1]
     
    X_t = test_set[0]
    y_t = test_set[1]
      
    X   = addOnes(X)
    X_v = addOnes(X_v)
    X_t = addOnes(X_t)
     
    y = mapClasses(y)
    y_v = mapClasses(y_v)
    y_t = mapClasses(y_t)
     
    del train_set,valid_set,test_set #release memory
     
#     alpha = 0.001
#     num_iters = 10
#     batchSize = 200
#     lambda_ = 0
        #batchSize = 50000
    method = "stGrad"   
    w = miniBatchGradientDescent(X,y,X_v,y_v,X_t,y_t,alpha,num_iters,batchSize,lambda_,method)

 
#error_train,error_validation,trErrorRate,valErroRate = \
  #trainRegression(X,y,X_v,y_v,X_t,y_t,lambda_)