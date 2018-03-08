#Assemble global stiffness matrix using element contributions
import numpy as np
from numba import jit
from Local_Matrices import MatrixT,MatrixK,MatrixS,VectorF
from getElementCoordinates import getElementCoordinates
from elements import Element
from scipy.sparse import *
from itertools import accumulate


def assembleTKandF(NodalCoord,AngularCoords,Connectivity,Coefficients,sourceFunc,el_type,Upwinded):
    if el_type=='Q4':
        num_node = 4
    elif el_type=='Q9':
        num_node = 9
    #now lets get the degrees of freedom (i.e. num_dof)
    num_dof = len(NodalCoord)
    num_el = len(Connectivity)
    num_dof_el = num_node #elemental degrees of freedom
    A_con = Connectivity
    A_i = np.zeros(16*num_el, dtype = int).transpose()
    A_j = np.zeros(16*num_el, dtype = int).transpose()
    A_v = np.zeros(16*num_el, dtype = float).transpose()
    F_i = np.zeros(4*num_dof, dtype = float).transpose()
    F_j = np.zeros(4*num_dof, dtype = float).transpose()
    F_v = np.zeros(4*num_dof, dtype = float).transpose()
    ii = [0, 1, 2, 3, 0 ,1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    jj = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    ii_small = [0,1,2,3]
    kk = list(range(16))
    k_small = list(range(4))
    for e in range(len(Connectivity)):
        Coord_mat_el = getElementCoordinates(e,NodalCoord,Connectivity,el_type)
        T_e = MatrixT(Coord_mat_el,AngularCoords,Coefficients,el_type,Upwinded)
        K_e = MatrixK(Coord_mat_el,AngularCoords,Coefficients,el_type,Upwinded)
        S_e = MatrixS(Coord_mat_el,AngularCoords,Coefficients,el_type,Upwinded)
        F_e = VectorF(Coord_mat_el,AngularCoords,Coefficients,sourceFunc,Upwinded)
        A_i[kk] = np.array(A_con[e,ii]).astype(int)-1
        A_j[kk] = np.array(A_con[e,jj]).astype(int)-1
        A_v[kk] = T_e.flatten('F') + K_e[:].flatten('F') + S_e[:].flatten('F') #Flatten column wise
        F_i[k_small] = np.array(A_con[e,ii_small]).astype(int)-1
        F_v[k_small] = F_e[:].transpose()
        kk = [i+16 for i in kk]
        k_small = [i+4 for i in k_small]
        #for i in range(num_dof_el):
        #    dof_1 = int(A_con[e][i])-1 # get degrees of freedom
        #    F[dof_1,0] += F_e[i,0]
        #    for j in range(num_dof_el):
        #        dof_2 = int(A_con[e][j])-1
                #T[dof_1, dof_2] += T_e[i,j]
                #K[dof_1, dof_2] += K_e[i,j]
                #S[dof_1, dof_2] += S_e[i,j]
                # end j loop
            #end i loop
    #end elemental loop e
    #A = csr_matrix((A_v,(A_i,A_j)))
    F = csr_matrix((F_v,(F_i,F_j)))
    return A_i,A_j,A_v,F
