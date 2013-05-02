'''
Created on Apr 17, 2013

@author: gena
'''
'''
Created on Apr 15, 2013

@author: gena
'''
from nnNetwork import *
import cPickle,gzip

print "download data"
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

ind1 = range(0,np.size(train_set[0],0))
ind2 = range(0,np.size(valid_set[0],0))
ind3 = range(0,np.size(test_set[0],0))

np.random.shuffle(ind1) 
np.random.shuffle(ind2) 
np.random.shuffle(ind3) 

X = train_set[0][ind1]
y = train_set[1][ind1]
 
X_v = valid_set[0][ind2]
y_v = valid_set[1][ind2]
 
X_t = test_set[0][ind3]
y_t = test_set[1][ind3]

#X = featureNormilize(X)
#X_v = featureNormilize(X_v)
#X_t = featureNormilize(X_t)

print "add units"  
X   = addOnes(X)
X_v = addOnes(X_v)
X_t = addOnes(X_t)
 
print "map classes" 
y = mapClasses(y)
y_v = mapClasses(y_v)
y_t = mapClasses(y_t)
 
lambda_ = np.array([0])
 
del train_set,valid_set,test_set #release memory

hidden_layer_size = 380
alpha = 0.1
num_iters = 1000
batchSize = 1000
 
w = train(sigmoid,X,y,X_v,y_v,X_t,y_t,\
          hidden_layer_size, alpha, num_iters,batchSize,lambda_)