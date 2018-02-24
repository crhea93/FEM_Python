#Assemble global stiffness matrix using element contributions
import numpy as np
from Local_Matrices import getStiffnessmatrix,getMassMatrix,getInternalForces
from getElementCoordinates import getElementCoordinates,getElementDisplacements,getElementCoordinatesIndex


def assemble(NodalCoord,Connectivity,D,D_prime,thickness,epsilon,sigma,el_type='Q4',P_type='vector',Prob='ES'):
    num_node = 4
    #now lets get the degrees of freedom (i.e. num_dof)
    if (P_type=='vector'):
        num_dof = num_node * 2  # elemental dof
        num_dof_total = 2 * len(NodalCoord)
        A = np.zeros((len(Connectivity),num_dof)) #create assembly matrix which is num_elements x num_dof
        for e in range(len(Connectivity)): #for each element
            for i in range(num_node): #all the columns of a row
                for j in range(2): #num of dof per node
                    if j==0:
                        A[e,(i)*2+j] = 2*Connectivity[e][i]-1 #assign x first based of nodal id of certain element hence
                    # con_mat[e,i]
                    else:
                        A[e,(i)*2+j] = 2*Connectivity[e][i] #assign y from nodal ID

    #print(A)
    K = np.zeros((num_dof_total, num_dof_total))  # K matrix will be num_dof x num_dof
    Fsig = np.zeros((num_dof_total,1))
    M = np.zeros((num_dof_total,num_dof_total)) # Mass matrix
    #Fext = np.zeros((num_dof_total,1))
    for e in range(len(Connectivity)):
        Coord_mat_el = getElementCoordinates(e,NodalCoord,Connectivity)
        Coord_mat_ind = getElementCoordinatesIndex(e,Connectivity)
        epsilon_el = epsilon[:, 4*e:4*e+4]
        sigma_el = sigma[:, 4*e:4*e+4]
        K_e = getStiffnessmatrix(Coord_mat_el,D,thickness,epsilon_el)
        Fsig_e = getInternalForces(Coord_mat_el,sigma_el)
        M_e = getMassMatrix(Coord_mat_el)
        for i in range(num_dof):
            dof_1 = int(A[e][i])-1 # get degrees of freedom
            Fsig[dof_1,0] += Fsig_e[i,0]
            #Fext[dof_1,0] += Fext_e[i,0]
            for j in range(num_dof):
                dof_2 = int(A[e][j])-1
                K[dof_1, dof_2] += K_e[i,j]
                M[dof_1,dof_2] += M_e[i,j]
            # end j loop
        #end i loop
    #end elemental loop
    return K,A,Fsig,M
