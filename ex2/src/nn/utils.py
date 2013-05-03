'''
Created on Apr 19, 2013

@author: gena

'''
import numpy as np

def featureNormilize(X):
    mu = np.mean(X,0)
    std = np.std(X,0)
    return (X-mu)/std
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return 2*sigmoid(2*x) - 1

def tanhGradient(x):
    return 1 - np.power(tanh(x),2) 

def sigmoidGradient(x):
    return np.multiply(sigmoid(x),1 - sigmoid(x)) 

def softplus(X):
    return np.log(1 + np.exp(X))

def softPlusGrad(x):
    return sigmoid(x)        

def addOnes(X):
    return np.c_[np.ones(len(X)),X]

def mapClasses(y):
    "map classes to {0,1} representation"
    u = np.unique(y);
    bY = np.zeros((y.size,u.size))
    for i in range(0,y.size):
        bY[i][y[i]-1] = 1
    return  bY

def createBatches(X,y,batchSize):
    """
    Create batches with the given size
    retuns the list of the objects
    """
    n1,m1 = np.shape(X)
    n2,m2 = np.shape(y)

    if (n1<=batchSize):
        numBatches = 1
        arrX = X.reshape((numBatches,n1,m1))
        arrY = y.reshape((numBatches,n2,m2))

        arrX = list(arrX)
        arrY = list(arrY)
    else:
        numBatches = np.floor(n1/batchSize)
        bEl = batchSize*numBatches
        arrX = X[0:bEl,:].reshape((numBatches,batchSize,m1))
        arrY = y[0:bEl,:].reshape((numBatches,batchSize,m2))

        arrX = list(arrX)
        arrY = list(arrY)

        res = n1 - bEl

        if (res>0): #add the the rest of the elements
            arrX.append(X[bEl:n1,:].reshape((res,m1)))
            arrY.append(y[bEl:n2,:].reshape((res,m2)))

    return arrX,arrY