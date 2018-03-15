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
        elif self.type == 'Q9':
            #print("N matrix properly loaded")
            N_matrix =  N_Q9(x, y)
            return N_matrix
        else:
            print("Please enter Q4")
            return 0

    def G(self,x,y,C):
        if self.type == 'Q4':
            # print("B matrix properly loaded")
            G_matrix = G_Q4(x, y, C)
            return G_matrix
        elif self.type == 'Q9':
            # print("B matrix properly loaded")
            G_matrix = G_Q9(x, y, C)
            return G_matrix
        else:
            print("Please enter Q4")
            return 0

    def G_ext(self,x,y,C):
        if self.type == 'Q4':
            # print("B matrix properly loaded")
            G_matrix = G_Q4_ext(x, y, C)
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
    N = np.zeros((1,4),dtype = float)
    N[0,0] = N_1(x,y)
    N[0,1] = N_2(x,y)
    N[0,2] = N_3(x,y)
    N[0,3] = N_4(x,y)
    return N

def G_Q4(x, y, C):
    Q = np.zeros((2, 4),dtype = float)
    Q[0, 0] = -(1. / 4) * (1 - y)
    Q[0, 1] = (1. / 4) * (1 - y)
    Q[0, 2] = (1. / 4) * (1 + y)
    Q[0, 3] = -(1. / 4) * (1 + y)
    Q[1, 0] = -(1. / 4) * (1 - x)
    Q[1, 1] = -(1. / 4) * (1 + x)
    Q[1, 2] = (1. / 4) * (1 + x)
    Q[1, 3] = (1. / 4) * (1 - x)
    J = Q@C
    detJ = np.linalg.det(J)
    J_inv = np.linalg.inv(J)
    GradN = J_inv@Q
    return GradN, detJ

def G_Q4_ext(x, y, C):
    Q = np.zeros((2, 4),dtype = float)
    Q[0, 0] = -(1. / 4) * (1 - y)
    Q[0, 1] = (1. / 4) * (1 - y)
    Q[0, 2] = (1. / 4) * (1 + y)
    Q[0, 3] = -(1. / 4) * (1 + y)
    Q[1, 0] = -(1. / 4) * (1 - x)
    Q[1, 1] = -(1. / 4) * (1 + x)
    Q[1, 2] = (1. / 4) * (1 + x)
    Q[1, 3] = (1. / 4) * (1 - x)
    J = Q@C
    detJ = np.linalg.det(J)
    J_inv = np.linalg.inv(J)
    GradN = J_inv@Q
    return GradN, detJ,Q,J_inv

##------------Q9-------------#
def N_1(x,y):
    return (1/4)*(x**2-x)*(y**2-y)

def N_2(x,y):
    return (1/4)*(x**2+x)*(y**2-y)

def N_3(x,y):
    return (1/4)*(x**2+x)*(y**2+y)

def N_4(x,y):
    return (1/4)*(x**2-x)*(y**2+y)

def N_5(x,y):
    return (1/2)*(x**2-x)*(1-y**2)

def N_6(x,y):
    return (1/2)*(1-x**2)*(y**2+y)

def N_7(x,y):
    return (1/2)*(x**2-x)*(1-y**2)

def N_8(x,y):
    return (1/2)*(1-x**2)*(y**2-y)

def N_9(x,y):
    return (x**2-1)*(y**2-1)



def N_Q9(x, y):
    N = np.zeros((1,9))
    N[0,0] = N_1(x,y)
    N[0,1] = N_2(x,y)
    N[0,2] = N_3(x,y)
    N[0,3] = N_4(x,y)
    N[0,4] = N_5(x,y)
    N[0,5] = N_6(x,y)
    N[0,6] = N_7(x,y)
    N[0,7] = N_8(x,y)
    N[0,8] = N_9(x,y)
    return N


def G_Q9(x, y, C):
    Q = np.zeros((2, 9))
    # x derivatives
    Q[0, 0] = (1. / 4) * (2*x-1) * (y**2-y)
    Q[0, 1] = (1. / 4) * (2*x+1) * (y**2-y)
    Q[0, 2] = (1. / 4) * (2*x+1) * (y**2+y)
    Q[0, 3] = (1. / 4) * (2*x-1) * (y**2+y)
    Q[0, 4] = (1. / 2) * (2*x+1) * (1-y**2)
    Q[0, 5] = (1. / 2) * (-2*x) * (y**2+y)
    Q[0, 6] = (1. / 2) * (2*x-1) * (1-y**2)
    Q[0, 7] = (1. / 2) * (-2*x) * (y**2-y)
    Q[0, 8] = (2*x) * (y**2-1)
    #y derivatives
    Q[1, 0] = (1. / 4) * (x**2-x) * (2*y-1)
    Q[1, 1] = (1. / 4) * (x**2+x) * (2*y-1)
    Q[1, 2] = (1. / 4) * (x**2+x) * (2*y+1)
    Q[1, 3] = (1. / 4) * (x**2-x) * (2*y+1)
    Q[1, 4] = (1. / 2) * (x**2+x) * (-2*y)
    Q[1, 5] = (1. / 2) * (1-x**2) * (2*y+1)
    Q[1, 6] = (1. / 2) * (x**2-x) * (-2*y)
    Q[1, 7] = (1. / 2) * (1-x**2) * (2*y-1)
    Q[1, 8] = (x**2-1) * (2*y)
    J = Q.dot(C)
    GradN = np.matmul(np.linalg.inv(J),(Q))
    detJ = np.linalg.det(J)
    return GradN, detJ

#---------------------------#
