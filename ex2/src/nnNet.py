'''
Created on May 3, 2013

@author: gena
'''

'''
Created on Apr 30, 2013

@author: gena
'''

from argparse import ArgumentParser,FileType
from nn.utils import addOnes,mapClasses
from nn.nnNetwork import miniBatchLearning
import cPickle,gzip

#constatnts
DEFAULT_ALPHA = 0.001
DEFAULT_LAMBDA = 0
DEFAULT_METHOD = 'rmsprop'
DEFAULT_PATH = '/home/gena/lab/data/mnist.pkl.gz'
DEFAULT_BATCH = 250
DEFAULT_NUM_ITERS = 15
DEFAULT_HIDDEN_LAYER_SIZE = 300
DEFAULT_ACTIVATION_FUNCTION = "sigmoid"

def _argparse():
    argparse = ArgumentParser(' Neural network module')
    argparse.add_argument('-lr','--alpha',type = float,default=DEFAULT_ALPHA) #learning rate
    argparse.add_argument('-reg','--lambda_',type = float,default = DEFAULT_LAMBDA) #regulazation parameter
    argparse.add_argument('-md','--method',type = str,default = DEFAULT_METHOD) #learning method
    argparse.add_argument('-bs', '--batch_size', type=int, default=DEFAULT_BATCH)
    argparse.add_argument('-n', '--num_iter', type=int, default=DEFAULT_NUM_ITERS)
    argparse.add_argument('-hl', '--hidden_layer_size', type=int, default=DEFAULT_HIDDEN_LAYER_SIZE)
    argparse.add_argument('-af', '--activation_function', type=str, default=DEFAULT_ACTIVATION_FUNCTION)
    argparse.add_argument('-i', '--path', type=str, default=DEFAULT_PATH)
    return argparse

def train(hidden_layer_size,activation_function,alpha,num_iters,batchSize,lambda_,method,path):
    #print "*************************" + path
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
     
    X = train_set[0]
    y = train_set[1]
     
    X_v = valid_set[0]
    y_v = valid_set[1]
     
    X_t = test_set[0]
    y_t = test_set[1]
      
    X   = addOnes(X)
    X_v = addOnes(X_v)
    X_t = addOnes(X_t)
     
    y = mapClasses(y)
    y_v = mapClasses(y_v)
    y_t = mapClasses(y_t)
     
    del train_set,valid_set,test_set #release memory
     
  
    print "Number of iterations: " + str(num_iters)
    print "Batch size: " + str(batchSize)
    print "Regularization parameter: " + str(lambda_)
    print "Learning rate: " +  str(alpha) 
    print "Hidden layer size: " + str(hidden_layer_size)
    print "Activation function: " + activation_function
    print "Training method: " + method  
       
    w = miniBatchLearning(activation_function,X,y,X_v,y_v,X_t,y_t,hidden_layer_size, alpha, num_iters,batchSize,lambda_,method)
      

def main(args):
    argp = _argparse().parse_args(args[1:])
    train(argp.hidden_layer_size,argp.activation_function,argp.alpha,argp.num_iter,argp.batch_size,argp.lambda_,argp.method,argp.path)
    
if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
    
    
    