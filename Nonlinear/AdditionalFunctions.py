import numpy as np

def cov_func(Coords,var):
    inner = np.zeros((2*len(Coords),2*len(Coords)))

    for i in range(len(Coords)):
        for j in range(len(Coords)):
            inner[i,j] = -abs(Coords[i,0]-Coords[j,0])/1-abs(Coords[i,1]-Coords[j,1])/1
    val = var**2*np.exp(inner)
    
    return val
