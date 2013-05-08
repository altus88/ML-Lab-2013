'''
Created on Apr 21, 2013

@author: gena
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
import sklearn.datasets

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

def addOnes(X):
    return np.c_[np.ones(len(X)),X]

def mapClasses(y,nClasses):
    "map classes to {0,1} representation"
    #u = np.unique(y);
    bY = np.zeros((y.size,nClasses))
    for i in range(y.size):
        bY[i][y[i]-1] = 1
    return  bY

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return 2*sigmoid(2*x) - 1

def tanhGradient(x):
    return 1 - np.power(tanh(x),2) 

def sigmoidGradient(x):
    return np.multiply(sigmoid(x),1 - sigmoid(x))    

def featureNormilize(X):
    mu = np.mean(X,0)
    std = np.std(X,0)
    return (X-mu)/std

def pca(X,nComponents):
    m = len(X)
    X-=np.mean(X,0)
    U,S,V =  np.linalg.svd(X)
    return np.dot(X,V[:,:nComponents])

def projectData(X,U,k):
    return np.dot(X,U[0:k,:])

def scatterPlot2D(X,y,nLabels,label_names,size = (50,50)):
   
    fig, plots = plt.subplots(nLabels, nLabels)
    fig.set_size_inches(size) 
    
    for i in range(nLabels):
        for j in range(nLabels):
            if i > j:
                continue
            x_  = X[(y == i) + (y == j)]
            y_  = y[(y == i) + (y == j)]
            
            X_transformed = pca(x_,2)
            plots[i, j].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
            plots[i, j].set_xticks(())
            plots[i, j].set_yticks(())
            
                        
            plots[j, i].scatter(X_transformed[:, 1], X_transformed[:, 0], c=y_)
            plots[j, i].set_xticks(())
            plots[j, i].set_yticks(())
            if i == 0:
                plots[i, j].set_title(label_names[j])
                plots[j, i].set_ylabel(label_names[j])
    #plt.tight_layout()   
    plt.tight_layout()
    plt.show()      
    raw_input("Press ENTER to exit")       
            
    