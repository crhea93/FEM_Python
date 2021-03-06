import numpy as np
from elements import Element
def MatrixT(Coords,Coefficients,Upwinded):
    E = Element('S2')
    Ke = np.zeros((2,2))
    #x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    #weights = [1,1]
    x_ip = [0]
    weights = [2]
    h = (Coords[1][0]-Coords[0][0])
    Jdet = h/2
    a = Coefficients[0]
    for i in range(len(x_ip)):
        N = E.N(x_ip[i])
        N_T = N.transpose()
        B = E.G(x_ip[i],Coords)
        B_T = B.transpose()
        Ke += a * N_T*B *Jdet * weights[i] + (h/2) * a * B_T * B * Jdet * weights[i];
    return Ke

def VectorF(Coords,Coefficients,sourceFunc,Upwinded):
    E = Element("S2")
    #x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    #weights = [1,1]
    x_ip = [0]
    weights = [2]
    f_ele = np.zeros((2,1))
    h = (Coords[1][0]-Coords[0][0])
    Jdet = h/2
    a = Coefficients[0]
    for i in range(len(x_ip)):
        N = E.N(x_ip[i])
        N_T = N.transpose()
        B = E.G(x_ip[i],Coords)
        B_T = B.transpose()
        x_coord = np.dot(np.array([Coords[:,0]]),N.transpose())
        source = SourceTerm(Coords,Coefficients,sourceFunc)
        f_ele +=  N_T*source* Jdet * weights[i] #+ (h/2)* a * B_T *source * Jdet * weights[i]
    return f_ele

def SourceTerm(coordinates,coefficients,sourceFunc):
    #This calculates the Source term
    source_term = 0
    x = coordinates[0][0]
    source_term +=  sourceFunc(x)
    return source_term
