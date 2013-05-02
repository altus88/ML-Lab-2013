from scipy.optimize import fmin_bfgs  ,fmin_l_bfgs_b
from numpy import *

def rosen_der(x,func):  
        xm = x[1:-1]
        xm_m1 = x[:-2]  
        xm_p1 = x[2:]  
        der = zeros(x.shape)  
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)  
        der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])  
        der[-1] = 200*(x[-1]-x[-2]**2)  
        return der
    
def costFunc(x,str1): 
    func = f
    
    return rosen_der(x,func)   
    
def grad(x,str1):
    func = f
    return rosen(x,func)   
    
def rosen(x,func):  # The Rosenbrock function  
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0) + func(0)    

def ff(f):
    return f(0)

def f(x):
    return x

x0 = [1.3, 0.7, 0.8, 1.9, 1.2]  
y = [1,2,3]

print ff(f)



xopt = fmin_l_bfgs_b(rosen, x0,None,(f,),1)
print xopt