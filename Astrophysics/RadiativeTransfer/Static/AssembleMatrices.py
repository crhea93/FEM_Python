#Assemble global stiffness matrix using element contributions
import numpy as np
from Local_Matrices import MatrixT,MatrixK,MatrixS,VectorF
from getElementCoordinates import getElementCoordinates
from elements import Element
from scipy.sparse import *
def assembleTKandF(NodalCoord,AngularCoords,Connectivity,Coefficients,sourceFunc,el_type,Upwinded):
    if el_type=='Q4':
        num_node = 4
    elif el_type=='Q9':
        num_node = 9
    #now lets get the degrees of freedom (i.e. num_dof)
    num_dof = len(NodalCoord)
    num_dof_el = num_node #elemental degrees of freedom
    A_con = Connectivity
    T = np.zeros((num_dof, num_dof))  # K matrix will be num_dof x num_dof
    K = np.zeros((num_dof, num_dof))
    S = np.zeros((num_dof, num_dof))
    F = np.zeros((num_dof,1))
    for e in range(len(Connectivity)):
        Coord_mat_el = getElementCoordinates(e,NodalCoord,Connectivity,el_type)
        T_e = MatrixT(Coord_mat_el,AngularCoords,Coefficients,el_type,Upwinded)
        K_e = MatrixK(Coord_mat_el,AngularCoords,Coefficients,el_type,Upwinded)
        S_e = MatrixK(Coord_mat_el,AngularCoords,Coefficients,el_type,Upwinded)
        F_e = VectorF(Coord_mat_el,AngularCoords,Coefficients,sourceFunc,Upwinded)
        for i in range(num_dof_el):
            dof_1 = int(A_con[e][i])-1 # get degrees of freedom
            F[dof_1,0] += F_e[i,0]
            for j in range(num_dof_el):
                dof_2 = int(A_con[e][j])-1
                T[dof_1, dof_2] += T_e[i,j]
                K[dof_1, dof_2] += K_e[i,j]
                S[dof_1, dof_2] += S_e[i,j]
                # end j loop
            #end i loop
    #end elemental loop e
    return K,T,S,F
