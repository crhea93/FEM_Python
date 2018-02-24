#Assemble global stiffness matrix using element contributions
import numpy as np
from Local_Matrices import Dif_basic_K,getStiffnessmatrix,getInternalForces,getdkdumatrix
from getElementCoordinates import getElementCoordinates,getElementDisplacements,getElementCoordinatesIndex


def assembleNL(NodalCoord,Connectivity,D,D_prime,thickness,epsilon,pt,sigma,el_type='Q4',P_type='vector',Prob='ES'):
    num_node = 4
    #now lets get the degrees of freedom (i.e. num_dof)
    if (P_type=='vector'):
        num_dof = num_node * 2  # elemental dof
        num_dof_total = 2 * len(NodalCoord)
        print("num_dof_total: ",num_dof_total)
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
    dk_du = np.zeros((num_dof_total, num_dof_total))
    Fsig = np.zeros((num_dof_total,1))
    #Fext = np.zeros((num_dof_total,1))
    for e in range(len(Connectivity)):
        Coord_mat_el = getElementCoordinates(e,NodalCoord,Connectivity)
        Coord_mat_ind = getElementCoordinatesIndex(e,Connectivity)
        epsilon_el = epsilon[:, 4*e:4*e+4]
        sigma_el = sigma[:, 4*e:4*e+4]
        K_e = getStiffnessmatrix(Coord_mat_el,D,thickness,epsilon_el,pt)
        dk_due = getdkdumatrix(Coord_mat_el,D_prime,thickness,epsilon_el)
        Fsig_e = getInternalForces(Coord_mat_el,sigma_el)
        #Fext_e = applyNeumann(Coord_mat_el,Coord_mat_ind,NBC,[1,0])
        for i in range(num_dof):
            dof_1 = int(A[e][i])-1 # get degrees of freedom
            Fsig[dof_1,0] += Fsig_e[i,0]
            #Fext[dof_1,0] += Fext_e[i,0]
            for j in range(num_dof):
                dof_2 = int(A[e][j])-1
                K[dof_1, dof_2] += K_e[i,j]
                dk_du[dof_1, dof_2] += dk_due[i,j]
            # end j loop
        #end i loop
    #end elemental loop
    return K,A,Fsig,dk_du
