'''
This module file will contain the different elements in class formats

Parameters:
    N - shape function matrix
    G - gradient matrix of shape functions
    H - laplacian matrix of shape functions
'''
import numpy as np


##class for element: will have element type. Based off that, we will assign it values for N,B, and H
class Element:
    def __init__(self, type):
        self.type = type

    def N(self,x):
        if self.type == 'S2':
            N_matrix =  N_S2(x)
            return N_matrix
        else:
            print("Please enter S2")
            return 0

    def G(self,x,Coords):
        if self.type == 'S2':
            G_matrix = G_S2(x,Coords)
            return G_matrix
        else:
            print("Please enter S2")
            return 0


def S_1(x):
    return (1/2)*(1-x)

def S_2(x):
    return (1/2)*(1+x)

def N_S2(x):
    N = np.zeros((1,2),dtype = float)
    N[0,0] = S_1(x)
    N[0,1] = S_2(x)
    return N


def G_S2(x,Coords):
    Q = np.zeros((1, 2),dtype = float)
    Jdet = (Coords[1][0]-Coords[0][0])/2
    Q[0,0] = -(1/2)*(1/Jdet)
    Q[0,1] = (1/2)*(1/Jdet)
    return Q
