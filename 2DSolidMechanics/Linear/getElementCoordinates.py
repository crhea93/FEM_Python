import numpy as np


def getElementCoordinates(elemNum, NodalCoord, Connectivity):
#This function returns the nodal coordinates of an element in the format
#C=[x1 y1; x2 y2, x3 y3;...]
    C = np.zeros((4,2))
    for i in range(0,4):
        node = Connectivity[elemNum][ i]-1

        C[i,0]=NodalCoord[node][ 0]
        C[i,1]=NodalCoord[node][ 1]
    return C

def getElementDisplacements(elemNum,d,A):
    vals = np.zeros((8,1))
    for i in range(0,4):
        dof1 = int( A[elemNum][2*i] - 1)
        dof2 = int( A[elemNum][2*i+1] - 1)
        vals[2*i,0] = d[dof1]
        vals[2*i+1,0] = d[dof2]
    return vals

def getElementCoordinatesIndex(e,Connectivity):
    C = np.zeros((4,1))
    for i in range(0,4):
        node = Connectivity[e][ i]-1

        C[i,0] = node
    return C
