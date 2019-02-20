
import matplotlib.pyplot as plt
import numpy as np
import random

x = np.arange(-100, 100, .50)

for i in range(1000):
    plt.figure(i)
    plt.xlabel('x')
    plt.ylabel('y')

    ax = plt.gca()

    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    j = random.uniform(-20000.01, 20000.01)
    plt.xlim(-100, 100)
    plt.ylim(-60000, 60000)
    plt.plot(x, j * x + 100, 'k')

#plt.show()
    plt.savefig('C:\\Users\\18732\Desktop\\autoencoder-Line-master\\line_picture\\' + str(i) + '.png')
    plt.close()