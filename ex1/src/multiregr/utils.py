'''
@author: Gennady Shabanov
'''
import numpy as np
import matplotlib.pyplot as plb
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import logsumexp

def logitPredict(X, w):
    """ Returns the ranking of the classes
    
    Parameters
    ----------
    X : array_like, shape = (n_samples,n_features)
        Input data.
    
    w : array_like, shape = (n_features,n_classes)
    
    Returns
    -------
    res: array_like, shape = (n_samples,n_classes)
        
    """
    Xw = np.dot(X, w);
    powe = np.power(np.e, Xw)
    den = np.atleast_2d(np.sum(powe,1))
    res = np.divide(powe,den.T)
    return res

def batchLogitPredict(X, w,nFeatures,nClasses,nBatches,batchSize):
    res = list()
    for i in range(0,nBatches):
        res.append(logitPredict(X[i],w))
    return res

def faceCostFunction(w,X,y,lambda_,nfeatures,nClasses,batchSize):
    w_ = np.reshape(w,(nfeatures,nClasses))
    return costFunction(X,y,w_,lambda_)

def faceGradient(w,X,y,lambda_,nfeatures,nClasses,batchSize):
    w_ = np.reshape(w,(nfeatures,nClasses))
    
    return  computeGrad(X,y,w_,lambda_).ravel().T 

def costFunction(X,y,w,lambda_):
    """
    Log-likelihood for multinomial logistic regression
    
    Parameters
    ----------
    X: array_like, shape = (n_samples,n_features)
       input data
       
    y: array_like (binary), shape = (n_samples,n_classes)
       classes 
    
    w : array_like, shape = (n_features,n_classes)
    
    lambda_: int
       Regularization parameter
    """
  
    Xw = np.dot(X,w)
    frstTerm = 0
    for i in range(0,length(X)):
        frstTerm = frstTerm + np.dot(X[i,:],w[:,np.nonzero(y[i,:])[0][0]])
    #frstTerm = np.sum(np.diag(np.dot(X,w[:,y.nonzero()[1]])))
    #scndTerm = np.sum(np.log(np.sum(np.power(np.e,Xw), axis=1)))
    scndTerm = np.sum(logsumexp(Xw,1))
    return  -(frstTerm - scndTerm) + np.sum(lambda_*np.sum(np.power(w,2)))

def batchCostFunction(X,y,w,lambda_,numBatches):
    cost = 0;
    for i in range(0,numBatches):
        cost+=costFunction(X[i],y[i],w,lambda_)
    return cost

def mapClasses(y):
    "map classes to {0,1} representation"
    u = np.unique(y);
    bY = np.zeros((y.size,u.size))
    for i in range(0,y.size):
        bY[i][y[i]-1] = 1
    return  bY

def length(X):
    if np.ndim(X)>1:
        return np.size(X,0)
    else:
        return 1

def addOnes(X):
    return np.c_[np.ones(length(X)),X]

def miniBatchGradientDescent(X,y,X_val,y_val,X_t,y_t, alpha, num_iters,batchSize,lambda_,method):
    """
    Performs the batch gradient descent to learn w
    alpha is the learning parameter
    w is  nfeatures x nclasses
    """	

    plb.figure(figsize = (10,8))
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
     
    cost_train = np.zeros((num_iters))

    error_train = np.zeros((num_iters))
    error_validation = np.zeros((num_iters))
    error_test = np.zeros((num_iters))
     
    X_,y_ = createBatches(X,y,batchSize)
    nBatches = len(X_)
     
    nClasses = np.size(y,1)
    nFeatures = np.size(X,1)
    w_ = np.zeros((nFeatures,nClasses))
    #w_ = initializeWeights(X,y)
      
    m = len(X_)
    
    def stGrad(X,y,w):
        g = computeGrad(X,y,w,lambda_)
        return w_- alpha*g  
    
    def l_bfgs_b(X,y,w):
        res = fmin_l_bfgs_b(faceCostFunction, w_.ravel(), faceGradient, \
                           (X,y,lambda_,nFeatures,nClasses,batchSize),\
                           approx_grad=0,bounds=None, m=10, factr=1e7, pgtol=1e-5,\
                           epsilon=1e-8,iprint=0, maxfun=15000, maxiter=10,\
                           disp=1, callback=None)
        return res[0]
    
    if (method == 'stGrad'):
        trFunction = stGrad
    elif (method == 'l_bfgs'):
        trFunction = l_bfgs_b
    
    for i in range(1,num_iters+1):
        for j in range(0,m):
            w_ = trFunction(X_[j],y_[j],w_)

        w_ = np.reshape(w_,(nFeatures,nClasses))
        cost_train[i-1] = batchCostFunction(X_,y_,w_,lambda_,m)
        
               
        print "[ "+str(i)+" ]"
        print "Cost " + str(cost_train[i-1])
        
         
        err1 = pedictionError(y_,\
                              batchLogitPredict(X_, w_,nFeatures,nClasses,nBatches,batchSize),nBatches,n_tr) 
        
        err2 = pedictionError(np.array(y_t,ndmin = 3),\
                              np.array(logitPredict(X_t, w_),ndmin = 3),1,n_test)
        
        err3 = pedictionError(np.array(y_val,ndmin = 3),\
                              np.array(logitPredict(X_val, w_),ndmin = 3),1,n_val)
        
        
        error_train[i-1] = err1
        error_validation[i-1] = err2
        error_test[i-1] = err3
        
        if i>1:
            pl1.plot(range(0,i),cost_train[0:i], 'r')
            pl2.plot(range(0,i),error_train[0:i], 'r')
            pl2.plot(range(0,i),error_validation[0:i], 'b')
            pl2.plot(range(0,i),error_test[0:i], 'g')
            plb.draw()
        
        print "Training error" + str(err1)
        print "Test error: "  +  str(err2)  
        print "Validation error:"  +  str(err3) 
       
    visualizeFilters(w_[1:,:])    
  

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


#def computeGrad(X,y,w,lambda_):
def computeGrad(X,y,w,lambda_):
    """
    Compute the predefined gradient for multinomial regression
    y is {0,1}
    lambda_ is regularization parameter
    """
    return np.dot(X.T,logitPredict(X,w)-y) + 2*lambda_*w

def initializeWeights(X,y):
    m = np.size(y,1);
    n = np.size(X,1)
    return np.random.rand(n,m)
    #return np.zeros((n,m))

def pedictionError(y,y_pred,numBatches,n):
    """
    compute the error rate
    y is the ground truth
    y_pred is prediction
    """
    sum_ = 0
    for i in range(0,numBatches):
        sum_ += np.sum(np.argmax(y[i],1) == np.argmax(y_pred[i],1))
    return 1 - float(sum_)/n



