import numpy as np
from numba import jit
from elements import Element
@jit
def MatrixT(Coords,CoordsAngular,Coefficients,el_type,Upwinded):
    E = Element(el_type)
    Te = np.zeros((len(Coords),len(Coords)))
    n = CalcNorm(CoordsAngular)
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    weights = [1, 1]
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            N = E.N(x_ip[i],x_ip[j])
            N_T = N.transpose()
            B,Jdet,G,J_inv = E.G_ext(x_ip[i],x_ip[j],Coords)
            B_T =  B.transpose()
            Te += N_T *np.dot(n,B)*Jdet*weights[i]*weights[j]
            if Upwinded == True:
                x_coord = np.array([Coords[:,0]])@N.transpose()
                y_coord = np.array([Coords[:,1]])@N.transpose()
                delta = CalcDelta(Coefficients,[x_coord,y_coord],n)
                Te += delta * np.dot(B_T,n.transpose())*np.dot(n,B) * Jdet* weights[i] * weights[j]
    return Te
@jit
def MatrixK(Coords,CoordsAngular,Coefficients,el_type,Upwinded):
    E = Element(el_type)
    Ke = np.zeros((len(Coords),len(Coords)))
    n = CalcNorm(CoordsAngular)
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    weights = [1, 1]
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            N = E.N(x_ip[i],x_ip[j])
            N_T = N.transpose()
            B,Jdet,G,J_inv = E.G_ext(x_ip[i],x_ip[j],Coords)
            B_T =  B.transpose()
            x_coord = np.array([Coords[:,0]])@N_T
            y_coord = np.array([Coords[:,1]])@N_T
            kappa_and_sigma = (Coefficients[0](x_coord,y_coord)+Coefficients[3](x_coord,y_coord))
            Ke += N_T *(kappa_and_sigma)*N*Jdet*weights[i]*weights[j]
            if Upwinded == True:
                delta = CalcDelta(Coefficients,[x_coord,y_coord],n)
                Ke += delta * np.dot(B_T,n.transpose()) * kappa_and_sigma * N * Jdet* weights[i] * weights[j]
    return Ke
@jit
def MatrixS(Coords,CoordsAngular,Coefficients,el_type,Upwinded):
    E = Element(el_type)
    Se = np.zeros((len(Coords),len(Coords)))
    n = CalcNorm(CoordsAngular)
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    weights = [1, 1]
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            N = E.N(x_ip[i],x_ip[j])
            N_T = N.transpose()
            B,Jdet,G,J_inv = E.G_ext(x_ip[i],x_ip[j],Coords)
            B_T =  B.transpose()
            x_coord = np.array([Coords[:,0]])@N_T
            y_coord = np.array([Coords[:,1]])@N_T
            sigma = Coefficients[3](x_coord,y_coord)
            Se +=  - N_T *(sigma)*N*Jdet*weights[i]*weights[j]
            if Upwinded == True:
                delta = CalcDelta(Coefficients,[x_coord,y_coord],n)
                Se +=  - delta * np.dot(B_T,n.transpose()) * sigma * N * Jdet* weights[i] * weights[j]
    return Se

def VectorF(Coords,CoordsAng,Coefficients,sourceFunc,Upwinded):
    E = Element("Q4")
    n = CalcNorm(CoordsAng)
    weights = [1, 1]
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    f_ele = np.zeros((4,1))
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            N = E.N(x_ip[i],x_ip[j])
            N_T = N.transpose()
            B,Jdet = E.G(x_ip[i],x_ip[j],Coords)
            B_T = B.transpose()
            x_coord = np.array([Coords[:,0]])@N.transpose()
            y_coord = np.array([Coords[:,1]])@N.transpose()
            source = SourceTerm([x_coord,y_coord],Coefficients,sourceFunc)
            f_ele +=  N_T*source* Jdet * weights[i] * weights[j]
            if Upwinded == True:
                x_coord = np.array([Coords[:,0]])@N.transpose()
                y_coord = np.array([Coords[:,1]])@N.transpose()
                delta = CalcDelta(Coefficients,[x_coord,y_coord],n)
                f_ele += delta*np.dot(B_T,n.transpose())*source* Jdet * weights[i] * weights[j]
    return f_ele

def CalcNorm(CoordinatesAngular,test_sl=False):
    norm = np.zeros((1,2),dtype = float)
    azimuthal = np.arctan(CoordinatesAngular[0,1]/CoordinatesAngular[0,0])
    theta = np.arccos(CoordinatesAngular[0,2]/np.sqrt(CoordinatesAngular[0,0]**2 + CoordinatesAngular[0,1]**2 + CoordinatesAngular[0,2]**2))
    norm[0,0] = theta
    norm[0,1] = azimuthal
    return norm


def CalcDelta(Coeff,Coord,norm):
    delta = 0.0
    x = Coord[0]
    y = Coord[1]
    kappa_val = Coeff[0](x,y)
    kappa_derx_val = Coeff[1](x,y)
    kappa_dery_val = Coeff[2](x,y)
    sigma_val = Coeff[3](x,y)
    if ((kappa_val!=0 and (kappa_derx_val**2+kappa_dery_val**2)!=0) and (sigma_val!=0)):
        v1 = (2*kappa_val)/(norm[0][0]*kappa_derx_val+norm[0][1]*kappa_dery_val)
        v2 = 1/sigma_val
        delta = min(v1,v2)
    else:
        delta = 0.01
    return delta

def SourceTerm(coordinates,coefficients,sourceFunc):
    source_term = 0
    x = coordinates[0]
    y = coordinates[1]
    source_term +=  sourceFunc(x,y)
    return source_term
