#Assemble global stiffness matrix using element contributions
import numpy as np
from numba import jit
from Local_Matrices import *
from getElementCoordinates import getElementCoordinates
from elements import Element
from scipy.sparse import *
from itertools import accumulate


def assemble(NodalCoord,norm,Connectivity,Coefficients,sourceFunc,el_type,Upwinded):
    num_node = Element(el_type).numnod
    #now lets get the degrees of freedom (i.e. num_dof)
    num_dof = len(NodalCoord)
    num_el = len(Connectivity)
    num_dof_el = num_node #elemental degrees of freedom
    A_con = Connectivity
    if el_type == 'Q4':
        num_per = 16
        small_num = 4
        ii = [0, 1, 2, 3, 0 ,1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        jj = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        ii_small = [0,1,2,3]
    if el_type == 'H8':
        num_per = 64
        small_num = 8
        ii = [0, 1, 2, 3, 4, 5 ,6 ,7 , 0, 1, 2, 3, 4, 5 ,6 ,7, 0, 1, 2, 3, 4, 5 ,6 ,7,0,1,2,3,4,5,6,7,0,1,2,3, 4, 5 ,6 ,7,0, 1, 2, 3, 4, 5 ,6 ,7,0, 1, 2, 3, 4, 5 ,6 ,7,0, 1, 2, 3, 4, 5 ,6 ,7 ]
        jj = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7]
        ii_small = [0,1,2,3,4,5,6,7]
    A_i = np.zeros(num_per*num_el, dtype = int).transpose()
    A_j = np.zeros(num_per*num_el, dtype = int).transpose()
    T_v = np.zeros(num_per*num_el, dtype = float).transpose()
    K_v = np.zeros(num_per*num_el, dtype = float).transpose()
    S_v = np.zeros(num_per*num_el, dtype = float).transpose()
    F_i = np.zeros(small_num*num_dof, dtype = float).transpose()
    F_j = np.zeros(small_num*num_dof, dtype = float).transpose()
    F_v = np.zeros(small_num*num_dof, dtype = float).transpose()
    kk = list(range(num_per))
    k_small = list(range(small_num))
    for e in range(len(Connectivity)):
        Coord_mat_el = getElementCoordinates(e,NodalCoord,Connectivity,el_type)
        if el_type == 'H8':
            T_e = MatrixT3D(Coord_mat_el,norm,Coefficients,el_type,Upwinded)
            K_e = MatrixK3D(Coord_mat_el,norm,Coefficients,el_type,Upwinded)
            S_e = MatrixS3D(Coord_mat_el,norm,Coefficients,el_type,Upwinded)
            F_e = VectorF3D(Coord_mat_el,norm,Coefficients,sourceFunc,el_type,Upwinded)
        if el_type == 'Q4':
            T_e = MatrixT2D(Coord_mat_el,norm,Coefficients,el_type,Upwinded)
            K_e = MatrixK2D(Coord_mat_el,norm,Coefficients,el_type,Upwinded)
            S_e = MatrixS2D(Coord_mat_el,norm,Coefficients,el_type,Upwinded)
            F_e = VectorF2D(Coord_mat_el,norm,Coefficients,sourceFunc,el_type,Upwinded)
        A_i[kk] = np.array(A_con[e,ii]).astype(int)-1
        A_j[kk] = np.array(A_con[e,jj]).astype(int)-1
        T_v[kk] = T_e[:].flatten('F')
        K_v[kk] = K_e[:].flatten('F')
        S_v[kk] = S_e[:].flatten('F') #Flatten column wise
        F_i[k_small] = np.array(A_con[e,ii_small]).astype(int)-1
        F_v[k_small] = F_e[:].transpose()
        kk = [i+num_per for i in kk]
        k_small = [i+small_num for i in k_small]
    F = csr_matrix((F_v,(F_i,F_j)))
    return A_i,A_j,T_v,K_v,S_v,F
