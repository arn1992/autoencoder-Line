
import matplotlib.pyplot as plt
import numpy as np
import random
from pylab import rcParams
import math
#x = np.arange(-100, 100, .50)

for i in range(0,361):
    #plt.figure(figsize=(1.7, 1.7))
    m=math.tan(math.radians(i))
    print(i,". ",m)

    x = np.arange(-100, 100, 5)
    #print(j)
    plt.xlabel('x')
    plt.ylabel('y')
    title = 'line_test' + str(i)


    ax = plt.gca()

    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    plt.xlim(-100, 100)
    plt.ylim(-60000, 60000)
    plt.plot(x,  m* x + 0,'k')
    #plt.savefig( 'D:/polynomial/line/data/movie/' +title + '.png')
    #plt.close()


plt.show()
