import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-100, 100, .50)

for i in range(500, 1000):
    plt.figure(i)
    plt.xlabel('x')
    plt.ylabel('y')

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    plt.xlim(-100, 100)
    plt.ylim(-40000, 40000)
    plt.plot(x, i*x)
    plt.savefig('C:\\Users\\18732\Desktop\\autoencoder-Line-master\\line_picture\\' + str(i) + '.png')
    plt.close()
