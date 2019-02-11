import matplotlib.pyplot as plt
import numpy as np
import random





x = np.arange(-100,100,5)

for i in range (500):
    title='line' + str(i)
    j=random.uniform(-100.01, 100.01)
    print(j)




    plt.plot(x, j*x+10.50)

    plt.savefig(title+".png")
    plt.close()


