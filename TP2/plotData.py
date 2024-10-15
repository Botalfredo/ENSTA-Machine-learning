import cmap as cmap
import numpy as np
from matplotlib import pyplot as plt

def plotData(X,y):

    cmap = 'bwr' # 'RdBu'
    cmap = plt.get_cmap(cmap)

    pos = X[(y == 1).flatten(), :]
    neg = X[(y == 0).flatten(), :]

    plt.figure()

    plt.scatter(neg[:, 0], neg[:, 1], s=20, marker='P', color=cmap(0))
    plt.scatter(pos[:, 0], pos[:, 1], s=20, marker='o', color=cmap(cmap.N))

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    plt.grid(True)
    plt.legend(['Admitted (y=1)', 'Not admitted (y=0)'], loc='upper right', shadow=True, fontsize = 'x-large', numpoints = 1)
    #plt.show()
