'''
Created on Apr 22, 2013

@author: gena
'''
import numpy as np
from utils import *
import time

def nnGrad(W,X,y,lambda_,sparsity_param,beta,func,nFeatures,hidden_layer_size,nClasses):
        """
            Computing the gradient by error back propagation
            f is the activation function
            f_grad is the gradient of the activation function
        """
        
        #start_time = time.time()
        w1 = np.reshape(W[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))
        
        w2 = np.reshape(W[(nFeatures+1)*hidden_layer_size:],\
                        ((hidden_layer_size+1),nClasses))
        
        # print str(elapsed_time) + " reshape"
        
        activation_function = func[0]
        grad = func[1]
            
        #start_time = time.time()
        m = float(len(X))
        DELTA_1 = np.zeros((nFeatures+1,hidden_layer_size))
        DELTA_2 = np.zeros((hidden_layer_size+1,nClasses))
        
        
        w1_grad = np.zeros((nFeatures+1,hidden_layer_size))
        w2_grad = np.zeros((hidden_layer_size+1,nClasses))
        
        delta_3 = np.zeros((1,nClasses))
        delta_2 = np.zeros((nFeatures+1,1))
        
        #startTime = time.time()
        
        z_2 = np.dot(X,w1)
        a_2 = addOnes(activation_function(z_2))
        z_3 = np.dot(a_2,w2)
        a_3 = activation_function(z_3)
        
        #activation penalty
        p_hat = np.mean(a_2[:,1:],0)
        penalty =  beta*(-(sparsity_param/p_hat) + (1-sparsity_param)/(1-p_hat))
        penalty = np.atleast_2d(np.concatenate(([0],penalty))).T
        
        
        delta_3 = a_3-y
        delta_2 = (np.dot(w2,delta_3.T)+ penalty)*grad(addOnes(z_2)).T
       
        for j in range(0,int(m)):
            DELTA_1 = DELTA_1 + np.dot(np.atleast_2d(delta_2[1:,j]).T,np.atleast_2d(X[j,:])).T
            DELTA_2 = DELTA_2 + np.dot(np.atleast_2d(a_2[j,:]).T,np.atleast_2d(delta_3[j,:]))        
            
        #DELTA_1 = sum(map(lambda x,y: np.dot(np.atleast_2d(x).T,np.atleast_2d(y)).T,delta_2[1:,:].T,X))
        #DELTA_2 = sum(map(lambda x,y: np.dot(np.atleast_2d(x).T,np.atleast_2d(y)),a_2,delta_3))   
        
        #add first column since we do not regularize it
        w1_grad[0,:] = DELTA_1[0,:]/m
        w2_grad[0,:] = DELTA_2[0,:]/m
        
        w1_grad[1:,:] = DELTA_1[1:,:]/m + (lambda_/m)*w1[1:,:]
        w2_grad[1:,:] = DELTA_2[1:,:]/m +  (lambda_/m)*w2[1:,:]
        
        grad = np.concatenate((w1_grad.ravel(),w2_grad.ravel()))

        #elapsedTime = time.time() - startTime
        #print elapsedTime  
        return grad
    
def nnCostFunction(W,X,y,lambda_,sparsity_param,beta,func,nFeatures,hidden_layer_size,nClasses):
        """
        Regularized cost function for one hidden layer neural network 
        
        W is the [1 x (nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses] 
                 weight matrix for all layers
        f is the activation function
        y is [nSamples x nClasses] {0,1} matrix
        lambda_ is regulatization parameter
        sparsity_param is desired average activation of hidden unit
        beta is weight of sparsity penalty term
        """  
        #startTime = time.time()
        w1 = np.reshape(W[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))
        
        w2 = np.reshape(W[(nFeatures+1)*hidden_layer_size:],\
                        ((hidden_layer_size+1),nClasses))
        
        f = func[0] #Take activation function
        
        z_2 = np.dot(X,w1)
        a_2 = addOnes(f(z_2))
        a_3 = f(np.dot(a_2,w2))
        
        #a_3_ = nnPredict(w1,w2,X,f)
                
        m = len(X)
        
        
        # regulatization term
        reg  = (lambda_/(2.*m))*(np.sum(np.power(w1[1:,:],2)) + np.sum(np.power(w2[1:,:],2)))
        
        # sparsity constraint on the hidden units
        p_hat = np.mean(a_2[:,1:],0)    #average activation per unit
        penalty = beta*np.sum(sparsity_param*np.log(sparsity_param/p_hat) \
                      + (1 - sparsity_param)*np.log((1-sparsity_param)/(1-p_hat)))
        
        
        res = 0
        for i in range(0,m):
            res+= -sum(np.dot(y[i,:],np.mat(np.log(a_3[i,:])).T) + np.dot((1-y[i,:]),np.log(1-a_3[i,:])))/m 
        
        #res = -(np.sum(map(np.dot,y,np.log(a_3))) + np.sum(map(np.dot,1-y,np.log(1-a_3))))/m
        
        #elapsedTime = time.time() - startTime
        #print elapsedTime
        return res + reg + penalty      

def nnPredict(w1,w2,X,f):
    """
    w1 is [(nFeatures+1) x hidden_layer_size] input layer weight matrix
    w2 is [(hidden_layer_size+1)*nClasses]
    f  is the activation function
    """
    res1 = f(np.dot(X,w1))
    res1 = addOnes(res1)
    
    return f(np.dot(res1,w2))   

def nnBatchPredict(X,W,f,nFeatures,hidden_layer_size,nClasses,nBatches,batchSize):
   
    w1 = np.reshape(W[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))
        
    w2 = np.reshape(W[(nFeatures+1)*hidden_layer_size:],\
                        ((hidden_layer_size+1),nClasses))
    
    res  =  list()
    for i in range(0,nBatches):
        res.append(nnPredict(w1,w2,X[i],f))
    
    return res

def pedictionError(y,y_pred,numBatches):
    """
    computes the mean square error
    y is the ground truth
    y_pred is prediction
    """
    err = 0
    for i in range(0,numBatches):
        err += np.mean(np.sum(0.5*np.power(y[i] - y_pred[i],2),1))
    return float(err)/numBatches

def visualizeFilters(W,fig_size=(8, 8),imap = 'Greys'):
    W = W - np.mean(W) #rescale
    l,m = np.shape(W)
    nCols = int(np.ceil(np.sqrt(m)))
    nRows = nCols
    diff = m - np.power(nCols,2)
    
    if diff != 0:
        if diff > 0: #we should add rows
            md = diff / nCols
            nRows = nRows + md + int((diff - md) > 0)
        else:
            md = np.abs(diff) / nCols
            nRows = nRows - md 
             
    pw=np.sqrt(l)
    ph = pw
    
    hs = 2
    ws = 2
    
    fig1 = plt.figure(num=None, figsize = fig_size, dpi=80, facecolor='w', edgecolor='k')
   
    arr = np.ones((nRows*ph + (nRows + 1)*hs,nCols*pw + (nCols+1)*ws))*-1
           
    k = 0
    for i in range(0,nRows):
        diff = m - nCols*(i+1)
        nC = int(diff>0)*nCols + (nCols - np.abs(diff))*int(diff<0)
        for j in range(0,nC):
            norm=np.max(np.abs(W[:,k]));
            p1 = i*ph + hs*(i+1)
            p2 = j*pw + ws*(j+1)
            arr[p1:p1+ph,p2:p2+pw] = np.reshape(W[:,k],(pw,ph))/norm
            k+=1
    
    plt.imshow(arr, cmap=imap)
    plt.tight_layout()
    plt.show(fig1)

    plt.savefig("filters.png")   
    #raw_input("Press ENTER to exit")  
    

