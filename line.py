import matplotlib.pyplot as plt
import numpy as np





x = np.arange(-100,100,.50)

for i in range (1000):


    plt.plot(x, i*x)




plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')
plt.show()
