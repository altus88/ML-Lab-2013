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
from nn.nnNetwork import miniBatchLearning,gradientCheck
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

#TODO check if nn works with any number of hidden layers

def _argparse():
    argparse = ArgumentParser(' Neural network module')
    argparse.add_argument('-lr','--alpha',type = float,default=DEFAULT_ALPHA,help = 'Learning rate (default:' + str(DEFAULT_ALPHA) + ')') #learning rate
    argparse.add_argument('-reg','--lambda_',type = float,default = DEFAULT_LAMBDA,help ='Regularization parameter (default:' + str(DEFAULT_LAMBDA) + ')') #regulazation parameter
    argparse.add_argument('-md','--method',type = str,default = DEFAULT_METHOD,help = 'Learning method {rmsprop,momentum,l_bfgs_b} (default:' + str(DEFAULT_METHOD) + ')') #learning method
    argparse.add_argument('-bs', '--batch_size', type=int, default=DEFAULT_BATCH,help = 'Batch size (default:' + str(DEFAULT_BATCH) + ')')
    argparse.add_argument('-n', '--num_iter', type=int, default=DEFAULT_NUM_ITERS,help = 'Maximum number of epoch (default: ' + str(DEFAULT_NUM_ITERS) + ')')
    argparse.add_argument('-hl', '--hidden_layer_size', type=int, default=DEFAULT_HIDDEN_LAYER_SIZE,help = 'Hidden layer size (default: ' + str(DEFAULT_HIDDEN_LAYER_SIZE) + ')')
    argparse.add_argument('-af', '--activation_function', type=str, default=DEFAULT_ACTIVATION_FUNCTION,help = 'Activation function {sigmoid,tanh,softplus} (default: ' + str(DEFAULT_ACTIVATION_FUNCTION) + ')')
    argparse.add_argument('-i', '--path', type=str, default=DEFAULT_PATH,help = 'Path to the mnist data (default: ' + str(DEFAULT_PATH) + ')')
    argparse.add_argument('-gc','--gradient_check', action='store_true', default=False,help = 'Gradient check')
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
    if argp.gradient_check:
        diff = gradientCheck()
        print "The norm of the difference between the analytical and numerical gradients is " + str(diff) 
    else:
        train(argp.hidden_layer_size,argp.activation_function,argp.alpha,argp.num_iter,argp.batch_size,argp.lambda_,argp.method,argp.path)
    
if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
    
    
    