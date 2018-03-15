import numpy as np


def getElementCoordinates(elemNum, NodalCoord, Connectivity,el_type):
#This function returns the nodal coordinates of an element in the format
#C=[x1 y1; x2 y2, x3 y3;...]
    if el_type=='Q4':
        C = np.zeros((4,2))
        ran = 4
    elif el_type=='Q9':
        ran = 9
        C = np.zeros((9,2))
    for i in range(0,ran):
        node = Connectivity[elemNum][ i]-1
        C[i,0]=NodalCoord[node][ 0]
        C[i,1]=NodalCoord[node][ 1]
    return C
