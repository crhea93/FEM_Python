import numpy as np
from numba import jit
from elements import Element

#-----------------------------2D-----------------------------------------------#
def MatrixT2D(Coords,n,Coefficients,el_type,Upwinded):
    E = Element(el_type)
    Te = np.zeros((len(Coords),len(Coords)))
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    weights = [1, 1]
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            N = E.N(x_ip[i],x_ip[j],0)
            N_T = N.transpose()
            B,Jdet,G,J_inv = E.G_ext(x_ip[i],x_ip[j],0,Coords)
            B_T =  B.transpose()
            Te += N_T *np.dot(n,B)*Jdet*weights[i]*weights[j]
            if Upwinded == True:
                x_coord = np.array([Coords[:,0]])@N.transpose()
                y_coord = np.array([Coords[:,1]])@N.transpose()
                delta = CalcDelta2D(Coefficients,[x_coord,y_coord],n)
                Te += delta * np.dot(B_T,n.transpose())*np.dot(n,B) * Jdet* weights[i] * weights[j]
    return Te
@jit
def MatrixK2D(Coords,n,Coefficients,el_type,Upwinded):
    E = Element(el_type)
    Ke = np.zeros((len(Coords),len(Coords)))
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    weights = [1, 1]
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            N = E.N(x_ip[i],x_ip[j],0)
            N_T = N.transpose()
            B,Jdet,G,J_inv = E.G_ext(x_ip[i],x_ip[j],0,Coords)
            B_T =  B.transpose()
            x_coord = np.array([Coords[:,0]])@N_T
            y_coord = np.array([Coords[:,1]])@N_T
            kappa_and_sigma = (Coefficients[0](x_coord,y_coord)+Coefficients[3](x_coord,y_coord))
            Ke += N_T *(kappa_and_sigma)*N*Jdet*weights[i]*weights[j]
            if Upwinded == True:
                delta = CalcDelta2D(Coefficients,[x_coord,y_coord],n)
                Ke += delta * np.dot(B_T,n.transpose()) * kappa_and_sigma * N * Jdet* weights[i] * weights[j]
    return Ke
@jit
def MatrixS2D(Coords,n,Coefficients,el_type,Upwinded):
    E = Element(el_type)
    Se = np.zeros((len(Coords),len(Coords)))
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    weights = [1, 1]
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            N = E.N(x_ip[i],x_ip[j],0)
            N_T = N.transpose()
            B,Jdet,G,J_inv = E.G_ext(x_ip[i],x_ip[j],0,Coords)
            B_T =  B.transpose()
            x_coord = np.array([Coords[:,0]])@N_T
            y_coord = np.array([Coords[:,1]])@N_T
            sigma = Coefficients[3](x_coord,y_coord)
            Se +=  - N_T *(sigma)*N*Jdet*weights[i]*weights[j]
            if Upwinded == True:
                delta = CalcDelta2D(Coefficients,[x_coord,y_coord],n)
                Se +=  - delta * np.dot(B_T,n.transpose()) * sigma * N * Jdet* weights[i] * weights[j]
    return Se

def VectorF2D(Coords,n,Coefficients,sourceFunc,El_type,Upwinded):
    E = Element(El_type)
    weights = [1, 1]
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    f_ele = np.zeros((4,1))
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            N = E.N(x_ip[i],x_ip[j],0)
            N_T = N.transpose()
            B,Jdet = E.G(x_ip[i],x_ip[j],0,Coords)
            B_T = B.transpose()
            x_coord = np.array([Coords[:,0]])@N_T
            y_coord = np.array([Coords[:,1]])@N_T
            source = SourceTerm2D([x_coord,y_coord],Coefficients,sourceFunc)
            f_ele +=  N_T*source* Jdet * weights[i] * weights[j]
            if Upwinded == True:
                delta = CalcDelta2D(Coefficients,[x_coord,y_coord],n)
                f_ele += delta*np.dot(B_T,n.transpose())*source* Jdet * weights[i] * weights[j]
    return f_ele

#-----------------------------3D-----------------------------------------------#
@jit
def MatrixT3D(Coords,n,Coefficients,el_type,Upwinded):
    E = Element(el_type)
    Te = np.zeros((len(Coords),len(Coords)))
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    weights = [1, 1]
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            for k in range(len(x_ip)):
                N = E.N(x_ip[i],x_ip[j],x_ip[k])
                N_T = N.transpose()
                B,Jdet,G,J_inv = E.G_ext(x_ip[i],x_ip[j],x_ip[k],Coords)
                B_T =  B.transpose()
                Te += N_T *np.dot(n,B)*Jdet*weights[i]*weights[j]*weights[k]
                if Upwinded == True:
                    x_coord = np.array([Coords[:,0]])@N.transpose()
                    y_coord = np.array([Coords[:,1]])@N.transpose()
                    z_coord = np.array([Coords[:,2]])@N.transpose()
                    delta = CalcDelta3D(Coefficients,[x_coord,y_coord,z_coord],n)
                    Te += delta * np.dot(B_T,n.transpose())*np.dot(n,B) * Jdet* weights[i] * weights[j]*weights[k]
    return Te
@jit
def MatrixK3D(Coords,n,Coefficients,el_type,Upwinded):
    E = Element(el_type)
    Ke = np.zeros((len(Coords),len(Coords)))
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    weights = [1, 1]
    x_0 = Coefficients[6]
    r_c = Coefficients[4]
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            for k in range(len(x_ip)):
                N = E.N(x_ip[i],x_ip[j],x_ip[k])
                N_T = N.transpose()
                B,Jdet,G,J_inv = E.G_ext(x_ip[i],x_ip[j],x_ip[k],Coords)
                B_T =  B.transpose()
                x_coord = np.array([Coords[:,0]])@N_T
                y_coord = np.array([Coords[:,1]])@N_T
                z_coord = np.array([Coords[:,2]])@N_T
                kappa_and_sigma = (Coefficients[0](x_coord,y_coord,z_coord)+Coefficients[3](x_coord,y_coord,z_coord,r_c,x_0))
                Ke += N_T *(kappa_and_sigma)*N*Jdet*weights[i]*weights[j]*weights[k]
                if Upwinded == True:
                    delta = CalcDelta3D(Coefficients,[x_coord,y_coord,z_coord],n)
                    Ke += delta * np.dot(B_T,n.transpose()) * kappa_and_sigma * N * Jdet* weights[i] * weights[j] * weights[k]
    return Ke
@jit
def MatrixS3D(Coords,n,Coefficients,el_type,Upwinded):
    E = Element(el_type)
    Se = np.zeros((len(Coords),len(Coords)))
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    weights = [1, 1]
    x_0 = Coefficients[6]
    r_c = Coefficients[4]
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            for k in range(len(x_ip)):
                N = E.N(x_ip[i],x_ip[j],x_ip[k])
                N_T = N.transpose()
                B,Jdet,G,J_inv = E.G_ext(x_ip[i],x_ip[j],x_ip[k],Coords)
                B_T =  B.transpose()
                x_coord = np.array([Coords[:,0]])@N_T
                y_coord = np.array([Coords[:,1]])@N_T
                z_coord = np.array([Coords[:,2]])@N_T
                sigma = Coefficients[3](x_coord,y_coord,z_coord,r_c,x_0)
                Se +=  - N_T *(sigma)*N*Jdet*weights[i]*weights[j]*weights[k]
                if Upwinded == True:
                    delta = CalcDelta3D(Coefficients,[x_coord,y_coord,z_coord],n)
                    Se +=  - delta * np.dot(B_T,n.transpose()) * sigma * N * Jdet* weights[i] * weights[j] *weights[k]
    return Se

def VectorF3D(Coords,n,Coefficients,sourceFunc,El_type,Upwinded):
    E = Element(El_type)
    weights = [1, 1]
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    f_ele = np.zeros((len(Coords),1))
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            for k in range(len(x_ip)):
                N = E.N(x_ip[i],x_ip[j],x_ip[k])
                N_T = N.transpose()
                B,Jdet = E.G(x_ip[i],x_ip[j],x_ip[k],Coords)
                B_T = B.transpose()
                x_coord = np.array([Coords[:,0]])@N_T
                y_coord = np.array([Coords[:,1]])@N_T
                z_coord = np.array([Coords[:,2]])@N_T
                source = SourceTerm3D([x_coord,y_coord,z_coord],Coefficients,sourceFunc)
                f_ele +=  N_T*source* Jdet * weights[i] * weights[j] * weights[k]
                if Upwinded == True:
                    delta = CalcDelta3D(Coefficients,[x_coord,y_coord,z_coord],n)
                    f_ele += delta*np.dot(B_T,n.transpose())*source* Jdet * weights[i] * weights[j] * weights[k]
    return f_ele
#------------------------------------------------------------------------------#

def CalcNorm2D(CoordinatesAngular,test_sl=False):
    norm = np.zeros((1,2),dtype = float)
    x = CoordinatesAngular[0]
    y = CoordinatesAngular[1]
    z = CoordinatesAngular[2]
    azimuthal = np.arctan(y/x)
    theta = np.arccos(z/np.sqrt(x**2 + y**2 + z**2))
    norm[0,0] = theta
    norm[0,1] = azimuthal
    return norm

def CalcNorm3D(CoordinatesAngular,test_sl=False):
    norm = np.zeros((1,3),dtype = float)
    x = CoordinatesAngular[0]
    y = CoordinatesAngular[1]
    z = CoordinatesAngular[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    if x!=0:
        azimuthal = np.arctan(y/x)
    if x==0:
        azimuthal = np.pi/2
    theta = np.arccos(z/r)
    norm[0,0] = theta
    norm[0,1] = azimuthal
    norm[0,2] = r
    return norm



def CalcDelta2D(Coeff,Coord,norm):
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

def CalcDelta3D(Coeff,Coord,norm):
    delta = 0.0
    x = Coord[0]
    y = Coord[1]
    z = Coord[2]
    r_c = Coeff[4]
    x_0 = Coeff[6]
    kappa_val = Coeff[0](x,y,z)
    kappa_derx_val = Coeff[1](x,y,z)
    kappa_dery_val = Coeff[2](x,y,z)
    sigma_val = Coeff[3](x,y,z,r_c,x_0)
    if ((kappa_val!=0 and (kappa_derx_val**2+kappa_dery_val**2)!=0) and (sigma_val!=0)):
        v1 = (2*kappa_val)/(norm[0][0]*kappa_derx_val+norm[0][1]*kappa_dery_val)
        v2 = 1/sigma_val
        delta = min(v1,v2)
    else:
        delta = 0.01
    return delta

def SourceTerm2D(coordinates,coefficients,sourceFunc):
    source_term = 0
    x = coordinates[0]
    y = coordinates[1]
    source_term +=  sourceFunc(x,y)
    return source_term

def SourceTerm3D(coordinates,coefficients,sourceFunc):
    source_term = 0
    x = coordinates[0]
    y = coordinates[1]
    z = coordinates[2]
    r_s = coefficients[5]
    source_term +=  sourceFunc(x,y,z,r_s)
    return source_term
