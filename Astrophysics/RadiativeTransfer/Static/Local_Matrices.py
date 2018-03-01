import numpy as np
from elements import Element
def MatrixT(Coords,CoordsAngular,Coefficients,el_type,Upwinded):
    E = Element(el_type)
    Ke = np.zeros((len(Coords),len(Coords)))
    n = CalcNorm(CoordsAngular)
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    weights = [1, 1]
    count = 0
    for x_iter in range(len(Coords)):
        for y_iter in range(len(Coords)):
            count = 0
            for i in range(len(x_ip)):
                for j in range(len(x_ip)):
                    N = E.N(x_ip[i],x_ip[j])
                    N_T = N.transpose()
                    B,Jdet,G,J_inv = E.G_ext(x_ip[i],x_ip[j],Coords)
                    #print(N_T[x_iter,:]@n)
                    n_dot_B = calc_dot(N_T[x_iter,:]@n,G,J_inv,y_iter)
                    Ke[x_iter,y_iter] += n_dot_B * Jdet * weights[i] * weights[j]
                    #Ke += N_T*calc_dot_trial(n,G,J_inv,count)*Jdet  * weights[i] * weights[j]
                    if Upwinded == True:
                        # We now need to calculate the global x,y positions for our qp
                        x_coord = np.array([Coords[:,0]])@N.transpose()
                        y_coord = np.array([Coords[:,1]])@N.transpose()
                        delta = .01#CalcDelta(Coefficients,[x_coord,y_coord],n)
                        test_dot = calc_dot_test(n.transpose(),G.transpose(),J_inv,y_iter)
                        #trial_dot = calc_dot_trial(n,G,J_inv,x_iter)
                        #Ke += delta*calc_dot_test*calc_dot_trial*Jdet
                        Ke[x_iter,y_iter] += delta *  calc_dot_trial(test_dot*n,G,J_inv,y_iter) * Jdet* weights[i] * weights[j]
                    count += 1
    return Ke


def VectorF(Coords,CoordsAng,Coefficients,sourceFunc,Upwinded):
    E = Element("Q4")
    n = CalcNorm(CoordsAng)
    weights = [1, 1]
    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]
    f_ele = np.zeros((4,1))
    count = 0
    for i in range(len(x_ip)):
        for j in range(len(x_ip)):
            N = E.N(x_ip[i],x_ip[j])
            N_T = N.transpose()
            B,Jdet = E.G(x_ip[i],x_ip[j],Coords)
            B_T = B.transpose()
            # We now need to calculate the global x,y positions for our qp
            x_coord = np.array([Coords[:,0]])@N.transpose()
            y_coord = np.array([Coords[:,1]])@N.transpose()
            source = SourceTerm([x_coord,y_coord],Coefficients,sourceFunc)
            f_ele +=  N_T*source* Jdet * weights[i] * weights[j]
            if Upwinded == True:
                # We now need to calculate the global x,y positions for our qp
                x_coord = np.array([Coords[:,0]])@N.transpose()
                y_coord = np.array([Coords[:,1]])@N.transpose()
                delta = CalcDelta(Coefficients,[x_coord,y_coord],n)
                f_ele += delta*B_T@n.transpose()*source* Jdet * weights[i] * weights[j]
            count +=1
    return f_ele

def CalcNorm(CoordinatesAngular):
    norm = np.zeros((1,2),dtype = float)
    norm[0,0] = np.arctan(CoordinatesAngular[1]/CoordinatesAngular[0])
    norm[0,1] = np.arctan(np.sqrt(CoordinatesAngular[0]**2 + CoordinatesAngular[1]**2 + CoordinatesAngular[2]**2)/CoordinatesAngular[2])
    return norm

def calc_dot(n,G,Jinv,count):
    dot_prod = 0
    dot_x = n[0]*(G[0,count]*Jinv[0,0]+G[1,count]*Jinv[0,1])
    dot_y = n[1]*(G[0,count]*Jinv[1,0]+G[1,count]*Jinv[1,1])
    dot_prod = dot_x+dot_y
    return dot_prod

def calc_dot_test(n,G,Jinv,count):
    dot_prod = 0
    dot_x = n[0,0]*(G[count,0]*Jinv[0,0]+G[count,1]*Jinv[0,1])
    dot_y = n[1,0]*(G[count,1]*Jinv[1,0]+G[count,1]*Jinv[1,1])
    dot_prod = dot_x+dot_y
    return dot_prod

def calc_dot_trial(n,G,Jinv,count):
    dot_prod = 0
    dot_x = n[0,0]*(G[0,count]*Jinv[0,0]+G[1,count]*Jinv[0,1])
    dot_y = n[0,1]*(G[0,count]*Jinv[1,0]+G[1,count]*Jinv[1,1])
    dot_prod = dot_x+dot_y
    return dot_prod

def CalcDelta(Coeff,Coord,norm):
    delta = 0
    #print(Coord)
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
    return delta

def SourceTerm(coordinates,coefficients,sourceFunc):
    #This calculates the Source term
    source_term = 0
    x = coordinates[0]
    y = coordinates[1]
    source_term +=  sourceFunc(x,y)
    return source_term
