import numpy as np


def getElementCoordinates(elemNum, NodalCoord, Connectivity):
    C = np.zeros((2,1))
    for i in range(2):
        node = Connectivity[elemNum][ i]
        C[i,0]=NodalCoord[node]
    return C
