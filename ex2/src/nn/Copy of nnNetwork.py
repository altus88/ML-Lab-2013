'''
Created on Apr 15, 2013

@author: gena
'''
import numpy as np
import scipy as sc
import pylab as plb
import time
from scipy.optimize import fmin_bfgs  ,fmin_l_bfgs_b


def nnGrad(W,X,y,lambda_,f,nFeatures,hidden_layer_size,nClasses):
        
        #start_time = time.time()
        w1 = np.reshape(W[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))
        
        w2 = np.reshape(W[(nFeatures+1)*hidden_layer_size:],\
                        ((hidden_layer_size+1),nClasses))
        
        #elapsed_time = time.time() - start_time 
        
        
       # print str(elapsed_time) + " reshape"
        
        
        start_time = time.time()
        m = len(X)
        DELTA_1 = np.zeros((nFeatures+1,hidden_layer_size))
        DELTA_2 = np.zeros((hidden_layer_size+1,nClasses))
        
        w1_grad = np.mat(np.zeros((nFeatures+1,hidden_layer_size)))
        w2_grad = np.mat(np.zeros((hidden_layer_size+1,nClasses)))
        
        delta_3 = np.zeros((1,nClasses))
        delta_2 = np.zeros((nFeatures+1,1))
        
        
    
        grad = np.zeros_like(W)
        for j in range(0,m):
            a_1 = X[j,:]
            z_2 = np.mat(np.dot(a_1,w1))
            a_2 = np.mat(f(z_2))
            a_2 = addOnes(a_2)
            z_3 = np.dot(a_2,w2)
            a_3 = f(z_3)
            delta_3 = (a_3-y[j,:])
            delta_2 = np.multiply(np.dot(np.mat(w2),delta_3.T),sigmoidGradient(addOnes(z_2)).T)
            DELTA_1 = DELTA_1 + np.dot(np.mat(delta_2[1:len(delta_2)]),np.mat(a_1)).T
            DELTA_2 = DELTA_2 + np.dot(a_2.T,delta_3)
        
        #add first column since we do not regularize it
        w1_grad[:,0] = DELTA_1[:,0]/m
        w2_grad[:,0] = DELTA_2[:,0]/m
        
        w1_grad[:,1:np.size(w1_grad,1)] = DELTA_1[:,1:np.size(DELTA_1,1)]/m+\
                                        (lambda_/m)*w1[:,1:np.size(w1,1)]
        w2_grad[:,1:np.size(w2,1)] = DELTA_2[:,1:np.size(DELTA_2,1)]/m+\
     (lambda_/m)*w2[:,1:np.size(w2,1)]
 
        grad = np.concatenate((w1_grad.ravel(),w2_grad.ravel()),1)
        
        elapsed_time = time.time() - start_time 
        print str(elapsed_time) + "Computing gradient"
        
        return np.array(grad)[0]
    
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
        m = len(X)
        
        reg  = (lambda_/(2.*m))*(np.sum(np.power(w1[1:,:],2)) + np.sum(np.power(w2[1:,:],2)))
        
        res = 0
        for i in range(0,m):
            res+= -sum(np.dot(y[i,:],np.mat(np.log(a_k[i,:])).T) + np.dot((1-y[i,:]),np.log(1-a_k[i,:])))/m 
        
        #Compute the gradient
#         DELTA_1 = np.zeros((nFeatures+1,hidden_layer_size))
#         DELTA_2 = np.zeros((hidden_layer_size+1,nClasses))
#          
#         w1_grad = np.mat(np.zeros((nFeatures+1,hidden_layer_size)))
#         w2_grad = np.mat(np.zeros((hidden_layer_size+1,nClasses)))
#      
#         for j in range(0,m):
#             a_1 = X[j,:]
#             z_2 = np.mat(np.dot(a_1,w1))
#             a_2 = np.mat(f(z_2))
#             a_2 = addOnes(a_2)
#             z_3 = np.dot(a_2,w2)
#             a_3 = f(z_3)
#             delta_3 = (a_3-y[j,:])
#             delta_2 = np.multiply(np.dot(np.mat(w2),delta_3.T),sigmoidGradient(addOnes(z_2)).T)
#             DELTA_1 = DELTA_1 + np.dot(np.mat(delta_2[1:len(delta_2)]),np.mat(a_1)).T
#             DELTA_2 = DELTA_2 + np.dot(a_2.T,delta_3)
#          
#         add first column since we do not regularize it
#         w1_grad[:,0] = DELTA_1[:,0]/m
#         w2_grad[:,0] = DELTA_2[:,0]/m
#          
#         w1_grad[:,1:np.size(w1_grad,1)] = DELTA_1[:,1:np.size(DELTA_1,1)]/m+\
#                                        (lambda_/m)*w1[:,1:np.size(w1,1)]
#         w2_grad[:,1:np.size(w2,1)] = DELTA_2[:,1:np.size(DELTA_2,1)]/m+\
#     (lambda_/m)*w2[:,1:np.size(w2,1)]
#   
#         grad = np.concatenate((w1_grad.ravel(),w2_grad.ravel()),1)
        
        #return (res + reg,np.array(grad)[0]) 
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return 2*sigmoid(2*x) - 1

def tanhGradient(x):
    return 1 - np.power(tanh(x),2) 

def sigmoidGradient(x):
    return np.multiply(sigmoid(x),1 - sigmoid(x))        

def addOnes(X):
    return np.c_[np.ones(len(X)),X]

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

    error_train = np.zeros((num_iters))
    error_validation = np.zeros((num_iters))
    error_test = np.zeros((num_iters))
     
    print "create batches" 
    X_,y_ = createBatches(X,y,batchSize)
    X_val_,y_val_  = createBatches(X_val,y_val,batchSize)
    X_t_,y_t_ = createBatches(X_t,y_t,batchSize)
    
    nClasses = np.size(y,1)
    nFeatures = np.size(X,1)-1
    #g = np.zeros((nFeatures,nClasses))
    #w_ = np.zeros(((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses,))
    
    w_ = np.random.rand((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses)/10000
      
    m = len(X_)
    cost_train = np.zeros((num_iters*m))
    
    #rmsprop parameters
    meanSqr = np.ones_like(w_)
    
    print "Run training"  
    #ind = 0
    trTime = 0
    #meanSqr = np.ones((m*num_iters,)) 
    
    for i in range(1,num_iters):
        #sf_val = 0
       
        for j in range(0,m):
            start_time = time.time()
            print "iter: " + str(j) + " epoch: " + str(i)
            g = nnGrad(w_,X_[j],y_[j],lambda_,f,nFeatures,hidden_layer_size,nClasses)
            meanSqr  = 0.9*meanSqr + 0.1*(g**2)
            w_ = w_- np.sqrt(meanSqr)*g
#             res = fmin_l_bfgs_b(nnCostFunction, w_, nnGrad, \
#                            (X_[j],y_[j],lambda_,f,nFeatures,hidden_layer_size,nClasses),0\
#                            ,m=10, factr=1e7, pgtol=1e-5,epsilon=1e-8,iprint=0,disp = 1,maxfun=15000, maxiter=50)
#             w_ = res[0]
            elapsed_time = time.time() - start_time 
            #trTime += elapsed_time
            print elapsed_time   
            print nnCostFunction(w_,X_[j],y_[j],lambda_,f,nFeatures,hidden_layer_size,nClasses)
#             sf_val +=f_val
#             if ((j+1) % 10 == 0): 
#                 cost_train[ind] = sf_val/10
#                 sf_val = 0
#                 ind+=1
#                 if j!=0:
#                     pl1.plot(range(0,ind),cost_train[0:ind], 'r')
#                     plb.draw()
            
        
            
            #w_ = w_.T
           
#             res = fmin_l_bfgs_b(nnCostFunction, w_.ravel(), None, \
#                           (X_[j],y_[j],lambda_,f,nFeatures,hidden_layer_size,nClasses),1\
#                           ,m=10, factr=1e7, pgtol=1e-5,epsilon=1e-8,iprint=0,disp = 1)
#             

        
        cost_train[i-1] = batchCostFunction(w_,X_,y_,lambda_,f,\
                                             nFeatures,hidden_layer_size,nClasses,m)
#         
        #cost_train[i-1] = sf_val/m
          
        #elapsed_time = time.time() - start_time
        #print "[ "+str(i)+" ]"
        print "cost: " + str(cost_train[i-1])
        
        #if mod(i,10) == 0:
        #start_time = time.time()
        
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
        
        elapsed_time = time.time() - start_time          
        print "Train error: " + str(err1) 
        print "Test error: " +  str(err2)  
        print "Validation error:" + str(err3)
        print "elapsed time for one epoch: " +  str(elapsed_time)

    raw_input("Press ENTER to exit")    
    return w_

def mapClasses(y):
    "map classes to {0,1} representation"
    u = np.unique(y);
    bY = np.zeros((y.size,u.size))
    for i in range(0,y.size):
        bY[i][y[i]-1] = 1
    return  bY

# def showReceptiveFields(w):
#     fig = plb.figure(2)
#     immap = 'Greys'
#     plb.subplot(2,5,1)
#     plb.imshow(reshape(w[1:,0],(28,28)), cmap=immap)
#     plb.subplot(2,5,2)
#     plb.imshow(reshape(w[1:,1],(28,28)), cmap=immap)
#     plb.subplot(2,5,3)
#     plb.imshow(reshape(w[1:,2],(28,28)), cmap=immap)
#     plb.subplot(2,5,4)
#     plb.imshow(reshape(w[1:,3],(28,28)), cmap=immap)
#     plb.subplot(2,5,5)
#     plb.imshow(reshape(w[1:,4],(28,28)), cmap=immap)
#     plb.subplot(2,5,6)
#     plb.imshow(reshape(w[1:,5],(28,28)), cmap=immap)
#     plb.subplot(2,5,7)
#     plb.imshow(reshape(w[1:,6],(28,28)), cmap=immap)
#     plb.subplot(2,5,8)
#     plb.imshow(reshape(w[1:,7],(28,28)), cmap=immap)
#     plb.subplot(2,5,9)
#     plb.imshow(reshape(w[1:,8],(28,28)), cmap=immap)
#     plb.subplot(2,5,10)
#     plb.imshow(reshape(w[1:,9],(28,28)), cmap=immap)
#     plb.show(fig)

# X = np.array([[1,1,2],[1,3,4]])
# y = np.array([[1, 0],[0, 1]])
# hl = 4
# nf = 2
# nClasses = 2
#  
# sw1 = (nf+1)*hl
# sw2 = (hl+1)*nClasses
# #w1 = np.ones((sw1,))
# #w2 = np.ones((sw2,))*2
# w1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
# w2 = np.array([13,14,15,16,17,18,19,20,21,22]) 
# 
# w = np.concatenate((w1,w2))
 
 

 
#print nnCostFunction(w,X,y,0,sigmoid,nf,hl,nClasses)





