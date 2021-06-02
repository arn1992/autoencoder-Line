import matplotlib.pyplot as plt
import numpy as np
import random





for i in range(1000):

    j = random.uniform(-10000.01, 10000.01)

    x = np.arange(-100, 100, 5)
    #print(j)
    plt.xlabel('x')
    plt.ylabel('y')
    title = 'line' + str(i)


    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    plt.xlim(-100, 100)
    plt.ylim(-60000, 60000)
    plt.plot(x, j*x+50)
    #plt.savefig( title + '.png')
    #plt.close()
plt.show()

