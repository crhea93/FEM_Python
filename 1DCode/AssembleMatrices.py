#Assemble global stiffness matrix using element contributions
import numpy as np
from Local_Matrices import MatrixT,VectorF
from getElementCoordinates import getElementCoordinates
from elements import Element

def assembleTandF(NodalCoord,Connectivity,Coefficients,sourceFunc,Upwinded):
    #now lets get the degrees of freedom (i.e. num_dof)
    num_dof = len(NodalCoord)
    num_dof_el = 2 #elemental degrees of freedom
    A = Connectivity
    K = np.zeros((num_dof, num_dof))  # K matrix will be num_dof x num_dof
    F = np.zeros((num_dof,1))
    for e in range(len(Connectivity)):
        Coord_mat_el = getElementCoordinates(e,NodalCoord,Connectivity)
        K_e = MatrixT(Coord_mat_el,Coefficients,Upwinded)
        F_e = VectorF(Coord_mat_el,Coefficients,sourceFunc,Upwinded)
        for i in range(num_dof_el):
            dof_1 = int(A[e][i])-1 # get degrees of freedom
            F[dof_1,0] += F_e[i,0]
            for j in range(num_dof_el):
                dof_2 = int(A[e][j])-1
                K[dof_1, dof_2] += K_e[i,j]
                # end j loop
            #end i loop
    #end elemental loop e
    return K,F
