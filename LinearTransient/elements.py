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

    # done initializing
    # now to set up the N object
    def N(self,x,y):
        if self.type == 'Q4':
            #print("N matrix properly loaded")
            N_matrix =  N_Q4(x, y)
            return N_matrix
        else:
            print("Please enter Q4")
            return 0

    def G(self,x,y):
        if self.type == 'Q4':
            #print("B matrix properly loaded")
            G_matrix = G_Q4(x, y)
            return G_matrix
        else:
            print("Please enter Q4")
            return 0

    def G_map(self,x,y,C):
        if self.type == 'Q4':
            # print("B matrix properly loaded")
            G_matrix = G_Q4_map(x, y, C)
            return G_matrix
        else:
            print("Please enter Q4")
            return 0
### Shape Functions assuming 1 1 on either side i.e. standard parents elements
#Q4

#B/c of cubit, swap (1 and 3) and (2 and 4)
def N_1(x,y):
    return (1/4)*(1-x)*(1-y)

def N_2(x,y):
    return (1/4)*(1+x)*(1-y)

def N_3(x,y):
    return (1/4)*(1+x)*(1+y)

def N_4(x,y):
    return (1/4)*(1-x)*(1+y)



def N_Q4(x, y):
    N = np.zeros((2, 8))
    N[0, 0] = N_1(x,y)
    N[0, 2] = N_2(x,y)
    N[0, 4] = N_3(x,y)
    N[0, 6] = N_4(x,y)
    N[1, 1] = N_1(x,y)
    N[1, 3] = N_2(x,y)
    N[1, 5] = N_3(x,y)
    N[1, 7] = N_4(x,y)


    return N
def G_Q4(x, y):
    Q = np.zeros((2, 4))
    Q[0, 0] = -(1. / 4) * (1 - y)
    Q[0, 1] = (1. / 4) * (1 - y)
    Q[0, 2] = (1. / 4) * (1 + y)
    Q[0, 3] = -(1. / 4) * (1 + y)
    Q[1, 0] = -(1. / 4) * (1 - x)
    Q[1, 1] = -(1. / 4) * (1 + x)
    Q[1, 2] = (1. / 4) * (1 + x)
    Q[1, 3] = (1. / 4) * (1 - x)
    return Q


def G_Q4_map(x, y, C):
    Q = np.zeros((2, 4))
    Q[0, 0] = -(1. / 4) * (1 - y)
    Q[0, 1] = (1. / 4) * (1 - y)
    Q[0, 2] = (1. / 4) * (1 + y)
    Q[0, 3] = -(1. / 4) * (1 + y)
    Q[1, 0] = -(1. / 4) * (1 - x)
    Q[1, 1] = -(1. / 4) * (1 + x)
    Q[1, 2] = (1. / 4) * (1 + x)
    Q[1, 3] = (1. / 4) * (1 - x)
    J = Q.dot(C)
    GradN = np.dot(np.linalg.inv(J),(Q))
    detJ = np.linalg.det(J)
    return GradN, detJ
