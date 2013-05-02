'''
Created on Apr 22, 2013

@author: gena
'''
import numpy as np
from utils import *

def nnGrad(W,X,y,lambda_,sparsity_param,beta,func,nFeatures,hidden_layer_size,nClasses):
        """
            Computing the gradient by error back propagation
            f is the activation function
            f_grad is the gradient of the activation function
        """
        
        w1 = np.reshape(W[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))
        
        w2 = np.reshape(W[(nFeatures+1)*hidden_layer_size:],\
                        ((hidden_layer_size+1),nClasses))
        
        f = func[0]
        f_grad = func[1]
   
        m = len(X)
        DELTA_1 = np.zeros((nFeatures+1,hidden_layer_size))
        DELTA_2 = np.zeros((hidden_layer_size+1,nClasses))
        
        
        w1_grad = np.mat(np.zeros((nFeatures+1,hidden_layer_size)))
        w2_grad = np.mat(np.zeros((hidden_layer_size+1,nClasses)))
        
        delta_3 = np.zeros((1,nClasses))
        delta_2 = np.zeros((nFeatures+1,1))
        
        #a1 = np.mat(X)
        z_2 = np.mat(np.dot(X,w1))
        a_2 = addOnes(np.mat(f(z_2)))
        a_3 = f(np.dot(a_2,w2))
        
        #activation penalty
        p_hat = np.mean(a_2[:,1:],0)
        penalty =  beta*(-(sparsity_param/p_hat) + (1-sparsity_param)/(1-p_hat))
        
              
        for j in range(0,m):
            a_1 = X[j,:]
            #z_2 = np.mat(np.dot(a_1,w1))
            #a_2 = np.mat(f(z_2))
            #a_2 = addOnes(a_2)
            #z_3 = np.dot(a_2,w2)
            #a_3 = f(z_3)
            delta_3 = (a_3[j,:]-y[j,:])
            delta_2 = np.multiply(np.dot(np.mat(w2),delta_3.T) + np.c_[0,penalty].T,f_grad(addOnes(z_2[j,:])).T)
            DELTA_1 = DELTA_1 + np.dot(np.mat(delta_2[1:len(delta_2)]),np.mat(a_1)).T
            DELTA_2 = DELTA_2 + np.dot(a_2[j,:].T,delta_3)
        
        #add first column since we do not regularize it
        w1_grad[:,0] = DELTA_1[:,0]/m
        w2_grad[:,0] = DELTA_2[:,0]/m
        
        w1_grad[:,1:np.size(w1_grad,1)] = DELTA_1[:,1:np.size(DELTA_1,1)]/m+\
                                        (lambda_/m)*w1[:,1:np.size(w1,1)]
        w2_grad[:,1:np.size(w2,1)] = DELTA_2[:,1:np.size(DELTA_2,1)]/m+\
     (lambda_/m)*w2[:,1:np.size(w2,1)]
 
        grad = np.concatenate((w1_grad.ravel(),w2_grad.ravel()),1)
        
          
        return np.array(grad)[0]
    
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
    
    res  =  np.zeros((nBatches,batchSize,nClasses))  #TODO implement the case if the number of batches do not split equally training samples
    for i in range(0,nBatches):
        res[i] = nnPredict(w1,w2,X[i],f)
    
    return res

def pedictionError(y,y_pred,numBatches,nSamples):
    """
    compute the error rate
    y is the ground truth
    y_pred is prediction
    """
    sum = 0
    for i in range(0,numBatches):
        sum += np.sum(np.argmax(y[i],1) == np.argmax(y_pred[i],1))
    return 1 - float(sum)/nSamples

def visualizeNet(W):
    W = W - np.mean(W) #rescale
    l,m = np.shape(W)
    nCols = int(np.sqrt(m))
    nRows = nCols
    
    pw=np.sqrt(l)
    ph = pw
    
    hs = 2
    ws = 2
    
    fig1 = plt.figure(num=None, figsize=(50, 50), dpi=80, facecolor='w', edgecolor='k')
   
    arr = np.ones((nRows*ph + (nRows + 1)*hs,nCols*pw + (nCols+1)*ws))
    immap = 'gray'
       
    k = 0
    for i in range(0,nRows):
        for j in range(0,nCols):
            norm=np.max(np.abs(W[:,k]));
            p1 = i*ph + hs*(i+1)
            p2 = j*pw + ws*(j+1)
            arr[p1:p1+ph,p2:p2+pw] = np.reshape(W[:,k],(pw,ph))/norm
            k+=1
    
    plt.imshow(arr, cmap=immap)
    plt.tight_layout()
    fig1.show()
    
    
#    fig, plots = plt.subplots(nRows,nCols)
#     fig.set_size_inches(200, 200)
#     immap = 'Greys'
#     k = 0
#     for i in range(0,nRows):
#         for j in range(0,nCols):
#             norm=np.max(np.abs(W[:,k]));
#             plots[i,j].imshow(np.reshape(W[:,k],(pw,ph))/norm, cmap=immap)
#             plots[i, j].set_xticks(())
#             plots[i, j].set_yticks(())
#             k+=1
# 
#     plt.tight_layout()
#    
    plt.savefig("filters.png")   
    raw_input("Press ENTER to exit")  
     
    
    
    #fig.set_size_inches(70, 70)
    

