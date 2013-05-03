'''
Created on Apr 15, 2013

@author: gena
'''
import numpy as np
import scipy as sc
import pylab as plb
import time
from scipy.optimize import fmin_bfgs  ,fmin_l_bfgs_b
from utils import *


def nnGrad(W,X,y,lambda_,f,nFeatures,hidden_layer_size,nClasses):
        
        start_time = time.time()
        w1 = np.reshape(W[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))
        
        w2 = np.reshape(W[(nFeatures+1)*hidden_layer_size:],\
                        ((hidden_layer_size+1),nClasses))
        
        # print str(elapsed_time) + " reshape"
            
        start_time = time.time()
        m = float(len(X))
        DELTA_1 = np.zeros((nFeatures+1,hidden_layer_size))
        DELTA_2 = np.zeros((hidden_layer_size+1,nClasses))
        
        
        w1_grad = np.zeros((nFeatures+1,hidden_layer_size))
        w2_grad = np.zeros((hidden_layer_size+1,nClasses))
        
        delta_3 = np.zeros((1,nClasses))
        delta_2 = np.zeros((nFeatures+1,1))
        
        #startTime = time.time()
        
        z_2 = np.dot(X,w1)
        a_2 = addOnes(f(z_2))
        z_3 = np.dot(a_2,w2)
        a_3 = f(z_3)
        
        delta_3 = a_3-y
        delta_2 = np.dot(w2,delta_3.T)*sigmoidGradient(addOnes(z_2)).T
        
        ##grad = np.zeros_like(W)
        #for j in range(0,int(m)):
#             a_1 = np.atleast_2d(X[j,:])
#             z_2 = np.atleast_2d(np.dot(a_1,w1))
#             a_2 = np.atleast_2d(f(z_2))
#             a_2 = addOnes(a_2)
#             z_3 = np.dot(a_2,w2)
#             a_3 = f(z_3)
            #delta_3 = (a_3-y[j,:])
            #delta_2 = np.dot(w2,delta_3.T)*sigmoidGradient(addOnes(z_2)).T
            #DELTA_1 = DELTA_1 + np.dot(np.atleast_2d(delta_2[1:,j]).T,np.atleast_2d(X[j,:])).T
            #DELTA_2 = DELTA_2 + np.dot(np.atleast_2d(a_2[j,:]).T,np.atleast_2d(delta_3[j,:]))
        
        #if batch size is not higher then 2000
        DELTA_1 = sum(map(lambda x,y: np.dot(np.atleast_2d(x).T,np.atleast_2d(y)).T,delta_2[1:,:].T,X))
        DELTA_2 = sum(map(lambda x,y: np.dot(np.atleast_2d(x).T,np.atleast_2d(y)),a_2,delta_3))
                
        #add first row since we do not regularize it
        w1_grad[0,:] = DELTA_1[0,:]/m
        w2_grad[0,:] = DELTA_2[0,:]/m
        
        w1_grad[1:,:] = DELTA_1[1:,:]/m + (lambda_/m)*w1[1:,:]
        w2_grad[1:,:] = DELTA_2[1:,:]/m +  (lambda_/m)*w2[1:,:]
 
        grad = np.concatenate((w1_grad.ravel(),w2_grad.ravel()),1)
        
        #elapsed_time = time.time() - start_time 
        #print str(elapsed_time) + " Computing gradient"
        
        return grad


    
def nnCostFunction(W,X,y,lambda_,f,nFeatures,hidden_layer_size,nClasses):
        """
        Regularized cost function for one hidden layer neural network 
        
        W is the [1 x (nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses] 
                 weight matrix for all layers
        f is the activation function
        y is [nSamples x nClasses] {0,1} matrix
        """  
        w1 = np.reshape(W[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))
        
        w2 = np.reshape(W[(nFeatures+1)*hidden_layer_size:],\
                        ((hidden_layer_size+1),nClasses))
        
        a_k = nnPredict(w1,w2,X,f)
        m = float(len(X))
        
        reg  = (lambda_/(2*m))*(np.sum(np.power(w1[1:,:],2)) + np.sum(np.power(w2[1:,:],2)))
        
        res = 0
        #for i in range(0,int(m)):
            #res+= -(np.dot(y[i,:],np.log(a_k[i,:]).T) + np.dot((1-y[i,:]),np.log(1-a_k[i,:])))/m 
        res = -(np.sum(map(np.dot,y,np.log(a_k))) + np.sum(map(np.dot,1-y,np.log(1-a_k))))/m
        return res + reg   
        
        
def batchCostFunction(W,X,y,lambda_,f,nFeautures,hidden_layer_size,nClasses,numBatches):
    cost = 0;
    for i in range(0,numBatches):
        cost+=nnCostFunction(W,X[i],y[i],lambda_,f,nFeautures,hidden_layer_size,nClasses)
    return cost/numBatches

    

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
    #TODO implement the case if the number of batches do not split equally training sample
    w1 = np.reshape(W[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))
        
    w2 = np.reshape(W[(nFeatures+1)*hidden_layer_size:],\
                        ((hidden_layer_size+1),nClasses))
    
    res  =  np.zeros((nBatches,batchSize,nClasses)) 
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
    

def train(f,X,y,X_val,y_val,X_t,y_t,hidden_layer_size, alpha, num_iters,batchSize,lambda_):
    """
    Performs the batch gradient descent to learn w
    alpha is the learning parameter
    w is  nfeatures x nclasses
    """    

    plb.figure(1)
    pl1 =  plb.subplot(211)
    pl2 =  plb.subplot(212)
    #pl3 =  plb.subplot(133)
    plb.ion()
    plb.show()
    
    n_val = np.size(X_val,0)
    n_test = np.size(X_t,0)
    n_tr = np.size(X,0)
     
     
    print "preallocate error arrays" 
    
#     error_train = list()
#     error_validation = list()
#     error_test = list()
    cost_train = np.zeros((num_iters))
    error_train = np.zeros((num_iters))
    error_validation = np.zeros((num_iters))
    error_test = np.zeros((num_iters))
     
    print "create batches" 
    X_,y_ = createBatches(X,y,batchSize)
    X_val_,y_val_  = createBatches(X_val,y_val,batchSize)
    X_t_,y_t_ = createBatches(X_t,y_t,batchSize)
    
    nClasses = np.size(y,1)
    nFeatures = np.size(X,1)-1
   
    #w_ = np.zeros(((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses,))
    
    w_ = np.array(np.random.rand((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses))/100
    g = np.zeros_like(w_)  
      
    m = len(X_)
       
    print "Run training"  
    
    def rmsprop(X,y,w):
        meanSqr = np.ones_like(w)
        for i in range(m):
            print "Iter: " + str(i)
            g = nnGrad(w,X[i],y[i],lambda_,f,nFeatures,hidden_layer_size,nClasses)
            meanSqr  = 0.9*meanSqr + 0.1*(np.power(g,2))
            w = w - alpha*(g/(np.sqrt(meanSqr) + 1e-8))
            print "Batch cost: " + str(nnCostFunction(w,X[i],y[i],lambda_,f,nFeatures,hidden_layer_size,nClasses))
        return w
    
    
    def momentum(X,y,w):
        w_diff = np.zeros_like(w)
        w_old  = np.zeros_like(w) 
        mu = 0.6
        w_old = w
        for i in range(m):
            g = nnGrad(w,X[i],y[i],lambda_,f,nFeatures,hidden_layer_size,nClasses)
            w = w- alpha*g + mu*(w_diff)
            w_old = w
            w_diff = w - w_old
            print nnCostFunction(w,X[i],y[i],lambda_,f,nFeatures,hidden_layer_size,nClasses)
        return w    
            
    def lbfs(X,y,w):
        for i in range(m):
            res = fmin_l_bfgs_b(nnCostFunction, w, nnGrad, \
                            (X[i],y[i],lambda_,f,nFeatures,hidden_layer_size,nClasses),0\
                           ,m=10, factr=1e7, pgtol=1e-5,epsilon=1e-8,iprint=0,disp = 1,maxfun=15000, maxiter=50)
            w = res[0]       
        return w
   
    for i in range(1,num_iters+1):
        print "epoch: " + str(i)
        w_ = rmsprop(X_,y_,w_)

        
        cost_train[i-1] = batchCostFunction(w_,X_,y_,lambda_,f,\
                                             nFeatures,hidden_layer_size,nClasses,m)
         
             
        #elapsed_time = time.time() - start_time
        #print "[ "+str(i)+" ]"
        print "cost: " + str(cost_train[i-1])
        
                
        err1 = pedictionError(y_,\
             nnBatchPredict(X_, w_,f,nFeatures,hidden_layer_size,nClasses,len(X_),batchSize),len(X_),n_tr)
          
         
        err2 = pedictionError(y_t_,\
             nnBatchPredict(X_t_, w_,f,nFeatures,hidden_layer_size,nClasses,len(X_t_),batchSize),len(X_t_),n_val)
         
        err3 = pedictionError(y_val_,\
             nnBatchPredict(X_val_, w_,f,nFeatures,hidden_layer_size,nClasses,len(X_val_),batchSize),len(X_val_),n_test)
              
        error_train[i-1] = err1
        error_validation[i-1] = err2
        error_test[i-1] = err3
        
        if i!=1:
            pl1.plot(range(0,i),cost_train[0:i], 'r')
         
            pl2.plot(range(0,i),error_train[0:i], 'r')
            pl2.plot(range(0,i),error_validation[0:i], 'b')
            pl2.plot(range(0,i),error_test[0:i], 'g')
         
            plb.draw()
         
        #elapsed_time = time.time() - start_time          
        print "Train error: " + str(err1) 
        print "Test error: " +  str(err2)  
        print "Validation error:" + str(err3)
        #print "elapsed time for one epoch: " +  str(elapsed_time)
    w1 = np.reshape(w_[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))    
    visualizeFilters(w1[1:,:])
    raw_input("Press ENTER to exit")    
    return w_



def visualizeFilters(W,fig_size=(8, 8)):
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
    
    fig1 = plb.figure(num=None, figsize = fig_size, dpi=80, facecolor='w', edgecolor='k')
   
    arr = np.ones((nRows*ph + (nRows + 1)*hs,nCols*pw + (nCols+1)*ws))*-1
    immap = 'Greys'
       
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
    
    plb.imshow(arr, cmap=immap)
    plb.tight_layout()
    plb.show(fig1)





