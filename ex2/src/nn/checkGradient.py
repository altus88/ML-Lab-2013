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

nSamples = 100
hidden_layer_size = 20
nFeatures = 15
nClasses = 10

X = np.random.rand(nSamples,nFeatures)
y = np.random.randint(nClasses,size = nSamples)
f = sigmoid
lambda_ = 0
w = np.random.rand((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses)
y = mapClasses(y)
X = addOnes(X)

grad = nnGrad(w, X, y, lambda_, f, nFeatures, hidden_layer_size, nClasses)
nmGrad = numGradient(cost(w),w)

#print grad
#print nmGrad

print  np.linalg.norm(nmGrad-grad)/np.linalg.norm(nmGrad+grad);
