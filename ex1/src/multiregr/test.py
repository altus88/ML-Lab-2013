import matplotlib.pyplot as plb
import numpy as np


plb.figure(figsize = (10,8), dpi=80, facecolor='w', edgecolor='k')
pl1 =  plb.subplot(211)
pl2 =  plb.subplot(212)

p1 = plb.Rectangle((0, 0), 0.1, 0.1, fc="r")
p2 = plb.Rectangle((0, 0), 1, 1, fc="g")


pl1.legend([p1,p2,p1], ["Red Rectangle","sad","sdsd"])
pl2.legend([p1], ["Red Rectangle",])

pl1.plot(range(0,10),np.random.rand(10), 'r',label = 'Test1')
pl1.plot(range(0,10),np.random.rand(10), 'g',label = 'Test2')
pl1.plot(range(0,10),np.random.rand(10), 'b',label = 'Test3')


#pl1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#plb.tight_layout()
plb.ion()
plb.show()
raw_input("Press ENTER to exit")  