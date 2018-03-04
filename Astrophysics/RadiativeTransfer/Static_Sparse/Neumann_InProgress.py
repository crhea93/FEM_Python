import numpy as np
from elements import Element
from getElementCoordinates import getElementCoordinates
def getForceFromGravity(NodalCoord, Connectivity, rho, g, h,el_type='Q4',P_type='vector'):
    if (el_type == 'Q4'):
        num_node = 4  # nodes per element
    elif (el_type == 'Q8'):
        num_node = 8
    else:
        print("Please enter a valid element type")

    # now lets get the degrees of freedom (i.e. num_dof)
    if (P_type == 'vector'):
        num_dof = num_node * 2  # elemental dof
        num_dof_total = 2 * len(NodalCoord)
        A = np.zeros((len(Connectivity), num_dof))  # create assembly matrix which is num_elements x num_dof
        for e in range(len(Connectivity)):  # for each element
            for i in range(num_node):  # all the columns of a row
                for j in range(2):  # num of dof per node
                    if j == 0:
                        A[e, (i) * 2 + j] = 2 * Connectivity[e][i] - 1  # assign x first based of nodal id of certain element hence
                        # con_mat[e,i]
                    else:
                        A[e, (i) * 2 + j] = 2 * Connectivity[e][i]  # assign y from nodal ID

    else:
        print("Please enter a valid problem type")

    F = np.zeros((num_dof_total,1))  # K matrix will be num_dof x num_dof

    for e in range(0, len(Connectivity)):
        #Coord_mat_el = getElementCoordinates(e, NodalCoord, Connectivity)
        # Loop over each element
        fe = F_GRAV([[0],[-rho*g]])*h
        for i in range(num_dof):
            dof_1 = int(A[e, i])-1 # get degrees of freedom and subtract 1 for now due to ordering
            F[dof_1, 0] += fe[i,0]

    # end elemental loop
    return F


def F_GRAV(trac):
    f_grav = np.zeros((8,1))
    E = Element("Q4")

    weights = [1, 1]
    xIP = [-np.sqrt(3) / 3, np.sqrt(3) / 3]
    for k in range(len(xIP)):
        s = xIP[k];
        N = E.N(s,s)
        J_s = np.matrix(detJsideQ4(s,s))
        n_mat = np.matrix(N.transpose())
        trac_mat =  np.matrix(trac)
        f_grav += n_mat*trac_mat*J_s[0,k]*weights[k]
    return f_grav

def detJsideQ4(x,y):
    G = np.array([-1/2,1/2])
    dxdpsi = G*x
    dydpsi = G*y
    detJside = np.sqrt(dxdpsi**2+dydpsi**2)
    return detJside
