'''
Created on Apr 18, 2013

@author: gena
'''

from nnNetwork import *
import numpy as np


def cost(w):
    return nnCostFunction(w, X, y, lambda_, f, nFeatures, hidden_layer_size, nClasses)

def numGradient(J,w):
    
    e = 0.0001
    p = np.zeros_like(w)
    grad_ = np.zeros_like(w)
    for i in range(np.size(w)):
        p[i] = e
        grad_[i] = np.divide(cost(w+p) - cost(w-p),2*e)
        p[i] = 0
    return grad_     

nSamples =  1000
hidden_layer_size =    15
nFeatures = 20
nClasses = 10

X = np.random.rand(nSamples,nFeatures)
y = np.random.randint(nClasses,size = nSamples)
f = sigmoid
lambda_ = 3
w = np.random.rand((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses)/100
y = mapClasses(y)
X = addOnes(X)

grad = nnGrad(w, X, y, lambda_, f, nFeatures, hidden_layer_size, nClasses)
#print grad
nmGrad = numGradient(cost(w),w)
#print cost(w)
#print grad
#print nmGrad

print  np.linalg.norm(nmGrad-grad)/np.linalg.norm(nmGrad+grad);
