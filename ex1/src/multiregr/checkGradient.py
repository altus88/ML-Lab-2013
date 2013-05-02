'''
Created on Apr 18, 2013

@author: gena
'''

from utils import *
import numpy as np
import time

def cost(w):
    w_ = np.reshape(w,(nFeatures+1,nClasses))
    return costFunction(X,y,w_,lambda_).flatten()

def grad(w):
    w_ = np.reshape(w,(nFeatures+1,nClasses))
    return computeGrad(X, y, w_, lambda_).flatten()

def numGradient(J,w):
    
    e = 0.0001
    p = np.zeros_like(w)
    grad_ = np.zeros_like(w)
    for i in range(np.size(w)):
        p[i] = e
        grad_[i] = np.divide(cost(w+p) - cost(w-p),2*e)
        p[i] = 0
    return grad_     

nSamples = 100
nFeatures = 15
nClasses = 10

X = np.random.rand(nSamples,nFeatures)
y = np.random.randint(nClasses,size = nSamples)

lambda_ = 5
w = np.random.rand((nFeatures+1)*nClasses)
y = mapClasses(y)
X = addOnes(X)

grad = grad(w)
nmGrad = numGradient(cost(w),w)

print  np.linalg.norm(nmGrad-grad)/np.linalg.norm(nmGrad+grad);
