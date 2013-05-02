from pylab import *
import numpy as np
import pylab as pl
import time

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)


def myplot():
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)
    
    figure(1)
    pl1 =  plt.subplot(211)
    pl2 =  plt.subplot(212)
    #pl3 =  plt.subplot(313)
    #plt.axis([0,3,0,1])
    #plt.a
    #plt.subplot(131)
    #plt.plot(t1, f(t1))
    
    
    plt.ion()
    show()
    i = 0
    while i<5:
               
        pl2.plot(t2[0:i], np.cos(2*np.pi*t2[0:i]), 'r--')
        pl1.plot(t1[0:i], f(t1[0:i]), 'bo')
        pl1.plot(t2[0:i], np.cos(2*np.pi*t2[0:i]), 'r--')
        i+=1
        plt.draw()
        time.sleep(0.05)
    
    
    fig =  figure(2)
    pl = plt.subplot(111)
    pl.plot(t2, np.cos(2*np.pi*t2), 'r--')  
    show(fig)
    
    raw_input("Press ENTER to exit")  
    return 0
z = myplot()

    

