
import matplotlib.pyplot as plt
import numpy as np
import random
from pylab import rcParams
#x = np.arange(-100, 100, .50)
j = 47000
for i in range(45,50):
    #plt.figure(figsize=(1.7, 1.7))
    print(i,". ",j)


    x = np.arange(-100, 100, .5)
    #print(j)
    plt.xlabel('x')
    plt.ylabel('y')
    title = 'line_test' + str(i)


    ax = plt.gca()

    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    plt.xlim(-100, 100)
    plt.ylim(-60000, 60000)
    plt.plot(x, j * x + 100,'k')
    #plt.savefig( 'D:/polynomial/line/data/movie/' +title + '.png')
    #plt.close()
    j=j+10000

plt.show()
