'''
Created on Apr 15, 2013

@author: gena
'''
import numpy as np
import pylab as plb
import time
from scipy.optimize import fmin_l_bfgs_b
from utils import *


def nnGrad(W,X,y,lambda_,func,nFeatures,hidden_layer_size,nClasses):
        """
          Compute the gradient using backpropagation
          
          Parameters
          ----------
          W: array-like, shape = ((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses))
             Weights of in the network
          
          X: array-like, shape = (nSamples,nFeatures+1)
             input data
          
          y: array-like, shape = (nSamples,nClasses)
             input data labels
             
          lambda_: int
             regularization parameter
           
          func: tuple
              func[0] - activation function
              func[1] - gradient of the activation function
                     
        """
        
        w1 = np.reshape(W[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))
        
        w2 = np.reshape(W[(nFeatures+1)*hidden_layer_size:],\
                        ((hidden_layer_size+1),nClasses))
        
        # print str(elapsed_time) + " reshape"
        
        activation_function = func[0]
        grad = func[1]
            
        #start_time = time.time()
        m = float(len(X))
        
        w1_grad = np.zeros((nFeatures+1,hidden_layer_size))
        w2_grad = np.zeros((hidden_layer_size+1,nClasses))
         
        z_2 = np.dot(X,w1)
        a_2 = addOnes(activation_function(z_2))
        z_3 = np.dot(a_2,w2)
        a_3 = activation_function(z_3)
        
        delta_3 = a_3-y
        delta_2 = np.dot(w2,delta_3.T)*grad(addOnes(z_2)).T
        
        if len(X)<=1000:
            #if batch size is not higher then 1000
            DELTA_1 = sum(map(lambda x,y: np.dot(np.atleast_2d(x).T,np.atleast_2d(y)).T,delta_2[1:,:].T,X))
            DELTA_2 = sum(map(lambda x,y: np.dot(np.atleast_2d(x).T,np.atleast_2d(y)),a_2,delta_3))
        else:
            DELTA_1 = np.zeros((nFeatures+1,hidden_layer_size))
            DELTA_2 = np.zeros((hidden_layer_size+1,nClasses))
            #delta_3 = np.zeros((1,nClasses))
            #delta_2 = np.zeros((nFeatures+1,1))
        ##grad = np.zeros_like(W)
            for j in range(0,int(m)):
                DELTA_1 = DELTA_1 + np.dot(np.atleast_2d(delta_2[1:,j]).T,np.atleast_2d(X[j,:])).T
                DELTA_2 = DELTA_2 + np.dot(np.atleast_2d(a_2[j,:]).T,np.atleast_2d(delta_3[j,:]))
            
                    
        #add first row since we do not regularize it
        w1_grad[0,:] = DELTA_1[0,:]/m
        w2_grad[0,:] = DELTA_2[0,:]/m
        
        w1_grad[1:,:] = DELTA_1[1:,:]/m + (lambda_/m)*w1[1:,:]
        w2_grad[1:,:] = DELTA_2[1:,:]/m +  (lambda_/m)*w2[1:,:]
 
        grad = np.concatenate((w1_grad.ravel(),w2_grad.ravel()),1)
        
        #elapsed_time = time.time() - start_time 
        #print str(elapsed_time) + " Computing gradient"
        
        return grad


    
def nnCostFunction(W,X,y,lambda_,func,nFeatures,hidden_layer_size,nClasses):
        """
        Regularized cost function for one hidden layer neural network 
        
        Parameters
          ----------
          W: array-like, shape = ((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses))
             Weights of in the network
          
          X: array-like, shape = (nSamples,nFeatures+)
             input data
          
          y: array-like, shape = (nSamples,nClasses)
             input data labels
             
          lambda_: int
             regularization parameter
           
          func: tuple
              func[0] - activation function
              func[1] - gradient of the activation function
        """ 
        f = func[0]
                
        w1 = np.reshape(W[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))
        
        w2 = np.reshape(W[(nFeatures+1)*hidden_layer_size:],\
                        ((hidden_layer_size+1),nClasses))
        
        a_k = nnPredict(w1,w2,X,f)
        m = float(len(X))
        
        reg  = (lambda_/(2*m))*(np.sum(np.power(w1[1:,:],2)) + np.sum(np.power(w2[1:,:],2)))
        
        res = 0
        
        if len(X) <= 1000:
            res = -(np.sum(map(np.dot,y,np.log(a_k))) + np.sum(map(np.dot,1-y,np.log(1-a_k))))/m
        else:
            for i in range(0,int(m)):
                res+= -(np.dot(y[i,:],np.log(a_k[i,:]).T) + np.dot((1-y[i,:]),np.log(1-a_k[i,:])))/m 
        
        return res + reg   
        
        
def batchCostFunction(W,X,y,lambda_,f,nFeautures,hidden_layer_size,nClasses,numBatches):
    cost = 0;
    for i in range(0,numBatches):
        cost+=nnCostFunction(W,X[i],y[i],lambda_,f,nFeautures,hidden_layer_size,nClasses)
    return cost/numBatches

    

def nnPredict(w1,w2,X,f):
    """Forward propagation through the network
    
    Parameters
    ----------
    w1 : array-like, shape = ((nFeatures+1) , hidden_layer_size)
        first layer weight matrix
    
    w2 :array-like,shape =  ((hidden_layer_size+1),nClasses)
        second layer weight matrix
        
    f  is the activation function
    """
    
    res1 = f(np.dot(X,w1))
    res1 = addOnes(res1)
    return f(np.dot(res1,w2))

def nnBatchPredict(X,W,func,nFeatures,hidden_layer_size,nClasses,nBatches,batchSize):
    
    w1 = np.reshape(W[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))
        
    w2 = np.reshape(W[(nFeatures+1)*hidden_layer_size:],\
                        ((hidden_layer_size+1),nClasses))
    
    res  =  list()
    f = func[0]
    for i in range(0,nBatches):
        res.append(nnPredict(w1,w2,X[i],f))
    
    return res
        

def pedictionError(y,y_pred,numBatches,nSamples):
    """Computes the error rate
    
    Parameters
    ----------
    y:array-like,shape = (nSamples,nClasses)
       the ground truth
    
    y_pred: array-like,shape = (nSamples,nClasses)
       predictions
       
    """
    sum = 0
    for i in range(0,numBatches):
        sum += np.sum(np.argmax(y[i],1) == np.argmax(y_pred[i],1))
    return 1 - float(sum)/nSamples
    

def miniBatchLearning(func,X,y,X_val,y_val,X_t,y_t,hidden_layer_size, alpha, num_iters,batchSize,lambda_,method):
    """ Performs mini-batch learning
    
    Parameters
    ----------
    X: array-like, shape = (nSamples,nFeatures+1)
        Training data
    
    y: array-like, shape = (nSamples,nClasses)
        Training labels
    
    X_val: array-like, shape = (nSamples,nFeatures+1)  
        Validation data
        
    y: array-like, shape = (nSamples,nClasses)
        Validation labels
        
    X_val: array-like, shape = (nSamples,nFeatures+1)  
        Test data
        
    y: array-like, shape = (nSamples,nClasses)
        Test labels  
        
    alpha: float
        The learning parameter
    
    method: str
        Learning method (rmsprop,momentum,lbfs)
        
    func: str
       Name of the activation function
    """    

    plb.figure(figsize = (10,8))

    plb.figure(1)
    pl1 =  plb.subplot(211)
    pl2 =  plb.subplot(212)
    
    p1 = plb.Rectangle((0, 0), 1, 1, fc="r")
    p2 = plb.Rectangle((0, 0), 1, 1, fc="g")
    p3 = plb.Rectangle((0, 0), 1, 1, fc="b")
    
    pl1.legend([p1], ["Cost function"])
    pl2.legend([p1,p2,p3], ["Training error rate","Test error rate","Validation error rate"])
    
    plb.ion()
    plb.show()
    
    
    n_val = np.size(X_val,0)
    n_test = np.size(X_t,0)
    n_tr = np.size(X,0)
     
    #preallocate error errays
    cost_train = np.zeros((num_iters))
    error_train = np.zeros((num_iters))
    error_validation = np.zeros((num_iters))
    error_test = np.zeros((num_iters))
     
    # create batches 
    X_,y_ = createBatches(X,y,batchSize)
    X_val_,y_val_  = createBatches(X_val,y_val,batchSize)
    X_t_,y_t_ = createBatches(X_t,y_t,batchSize)
    
    nClasses = np.size(y,1)
    nFeatures = np.size(X,1)-1
   
    #w_ = np.zeros(((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses,))
    w_ = np.array(np.random.rand((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses))/1000
    g = np.zeros_like(w_)  
      
    m = len(X_)
       
       
    if func == "sigmoid":
        f = (sigmoid,sigmoidGradient)
    elif func == "tanh":
        f = (tanh,tanhGradient) 
    elif func =="softplus":
        f = (softplus,softPlusGrad)
    else:
        raise Exception("Unknown activation function")
        #import sys
        #sys.exit()         
    
    def rmsprop(X,y,w):
        meanSqr = np.ones_like(w)
        for i in range(m):
            
            g = nnGrad(w,X[i],y[i],lambda_,f,nFeatures,hidden_layer_size,nClasses)
            meanSqr  = 0.9*meanSqr + 0.1*(np.power(g,2))
            w = w - alpha*(g/(np.sqrt(meanSqr) + 1e-8))
            print "Iter: " + str(i) + " Batch cost: " + str(nnCostFunction(w,X[i],y[i],lambda_,f,nFeatures,hidden_layer_size,nClasses))
        return w
    
    
    def momentum(X,y,w):
        w_diff = np.zeros_like(w)
        w_old  = np.zeros_like(w) 
        mu = 0.99
        w_old = w
        for i in range(m):
            g = nnGrad(w,X[i],y[i],lambda_,f,nFeatures,hidden_layer_size,nClasses)
            w = w- alpha*g + mu*(w_diff)
            w_old = w
            w_diff = w - w_old
            print "Iter: " + str(i) + " Batch cost: " + str(nnCostFunction(w,X[i],y[i],lambda_,f,nFeatures,hidden_layer_size,nClasses))
      
        return w    
            
    def l_bfgs_b(X,y,w):
        for i in range(m):
            res = fmin_l_bfgs_b(nnCostFunction, w, nnGrad, \
                            (X[i],y[i],lambda_,f,nFeatures,hidden_layer_size,nClasses),0\
                           ,m=10, factr=1e7, pgtol=1e-5,epsilon=1e-8,iprint=0,disp = 1,maxfun=15000, maxiter=30)
            w = res[0]       
            print "Iter: " + str(i) + " Batch cost: " + str(nnCostFunction(w,X[i],y[i],lambda_,f,nFeatures,hidden_layer_size,nClasses))
      
        return w
    
    if method == "rmsprop":
        training = rmsprop
    elif method == "momentum":
        training = momentum
    elif method == "l_bfgs_b":   
        training = l_bfgs_b
    else:
        raise Exception("Unknown training method")
        #import sys
        #sys.exit()    
     
    maxNumberVerrorIncrease = 10 #maximum number validation error increase        
    vErrorInc = 0
    
    bestVerr = np.Inf
    w_best = np.zeros_like(w_)
      
    for i in range(1,num_iters+1):
        print "epoch: " + str(i)
        w_ = training(X_,y_,w_)

        
        cost_train[i-1] = batchCostFunction(w_,X_,y_,lambda_,f,\
                                             nFeatures,hidden_layer_size,nClasses,m)
         
      
                           
        err1 = pedictionError(y_,\
             nnBatchPredict(X_, w_,f,nFeatures,hidden_layer_size,nClasses,len(X_),batchSize),len(X_),n_tr)
          
         
        err2 = pedictionError(y_t_,\
             nnBatchPredict(X_t_, w_,f,nFeatures,hidden_layer_size,nClasses,len(X_t_),batchSize),len(X_t_),n_val)
         
        err3 = pedictionError(y_val_,\
             nnBatchPredict(X_val_, w_,f,nFeatures,hidden_layer_size,nClasses,len(X_val_),batchSize),len(X_val_),n_test)
              
        error_train[i-1] = err1
        error_validation[i-1] = err2
        error_test[i-1] = err3
        
        if i>1:
            pl1.plot(range(0,i),cost_train[0:i], 'r')
            pl2.plot(range(0,i),error_train[0:i], 'r')
            pl2.plot(range(0,i),error_validation[0:i], 'b')
            pl2.plot(range(0,i),error_test[0:i], 'g')
            plb.draw()
            
        
        if i > 1: 
            if  error_validation[i-1] < bestVerr: #error rate is lower
                bestVerr = error_validation[i-1]
                w_best = w_
                if vErrorInc !=0: #zeroise counter
                    vErrorInc = 0
            else: #validation error is becoming worse
                vErrorInc +=1     
        else: #save layer and validation error on the first iterations
            bestVerr = error_validation[0]
            w_best = w_  
               
        #elapsed_time = time.time() - start_time          
        print "Cost function: " + str(cost_train[i-1])
        print "Train error: " + str(err1) 
        print "Test error: " +  str(err2)  
        print "Validation error:" + str(err3)
        
        if vErrorInc ==  maxNumberVerrorIncrease:
            break
        #print "elapsed time for one epoch: " +  str(elapsed_time)
    plb.savefig("plots.png")
    w1 = np.reshape(w_best[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))    
    
    visualizeFilters(w1[1:,:])
    return w_best


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
    plb.savefig("filters.png")  
    plb.show(fig1)
    

def gradientCheck():
    
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
    f = (sigmoid,sigmoidGradient)
    lambda_ = 3
    w = np.random.rand((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses)/100
    y = mapClasses(y)
    X = addOnes(X)
    
    grad = nnGrad(w, X, y, lambda_, f, nFeatures, hidden_layer_size, nClasses)
    #print grad
    nmGrad = numGradient(cost(w),w)
       
    return np.linalg.norm(nmGrad-grad)/np.linalg.norm(nmGrad+grad);




