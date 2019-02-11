import matplotlib.pyplot as plt
import numpy as np





x = np.arange(-360,360,4)

for i in range (500):
    title='line' + str(i)



    plt.plot(x, i*x)

    plt.savefig(title+".png")
    plt.close()


