import numpy as np
from scipy.sparse import *


def Apply_EBC(A_i,A_j,A_v,F,NodesCoord,BC_nodes,BC_vals,dictEBC):
    #first we need to see which nodes are not in the boundary!
    nodes = []
    for i in range(0,len(NodesCoord)):
        nodes.append(i)
    Nodes_wout_IC = []
    for i in range(len(nodes)):
        if (nodes[i] + 1 in list(dictEBC.keys())):
            pass
        else:
            Nodes_wout_IC.append(i)
    #now to fix F and K matrices
    num_wout = len(Nodes_wout_IC)
    num_with = len(dictEBC.keys())
    Ac_i = np.array([[]])
    Ac_j = np.array([[]])
    Ac_v = np.array([[]])
    F_c = np.zeros(num_wout, dtype = float)
    F = F.todense()
    A = csc_matrix((A_v,(A_i,A_j)))
    #Afull = A.todense()
    #A_c = np.zeros((num_wout,num_wout))



    nwo_count = 0
    for nwo in Nodes_wout_IC:
        nw_count = 0
        for nw in BC_nodes:
            nw = nw -1 #Gotta get rid of that pesky count at 1
            if nw_count==0: #if the first one affected
                F_c[nwo_count] = F[nwo] - dictEBC[BC_nodes[nw_count]]*A[nwo,nw]
            else: #all others based on what we already have for F_corrected just calculated
                F_c[nwo_count] -=  dictEBC[BC_nodes[nw_count]]*A[nwo,nw]
            nw_count += 1
        nwo_count += 1

    #nodes_without_and_in_A = list(set(A.indices).intersection(Nodes_wout_IC))

    #for k in range(len(Nodes_wout_IC)):
    #    row = Nodes_wout_IC[k]
    #    for c in range(len(Nodes_wout_IC)):
    #        col = Nodes_wout_IC[c]
    #        A_c[k,c] = Afull[row,col]
    #A_i_corrected = set(list(A_i).intersection(Nodes_wout_IC))
    ind_dict = dict((k,i) for i,k in enumerate(A_i))
    inter = set(ind_dict).intersection(Nodes_wout_IC)
    indices = [ ind_dict[x] for x in inter ]

    for it in range(len(indices)):
        current_index = indices[it]
        Ac_i = np.append(Ac_i,A_i[current_index])
        Ac_j = np.append(Ac_j,A_j[current_index])
        Ac_v = np.append(Ac_v,A_v[current_index])


    '''
    A_corrected = np.empty((len(Nodes_wout_IC), len(Nodes_wout_IC)))
    F_corrected = np.empty((len(Nodes_wout_IC),1))
    #Lets correct F first since it takes more work
    for i in range(len(Nodes_wout_IC)):
        current_node_without = Nodes_wout_IC[i]
        for j in range(len(BC_nodes)):
            current_node_with = BC_nodes[j]-1 #Gotta get rid of that pesky count at 1
            if j==0: #if the first one affected
                F_corrected[i] = F[current_node_without] - dictEBC[BC_nodes[j]]*A[current_node_without,current_node_with]
            else: #all others based on what we already have for F_corrected just calculated
                F_corrected[i] -=  dictEBC[BC_nodes[j]]*A[current_node_without,current_node_with]

    #And now T, which is really simple
    for k in range(len(Nodes_wout_IC)):
        row = Nodes_wout_IC[k]
        for c in range(len(Nodes_wout_IC)):
            col = Nodes_wout_IC[c]
            A_corrected[k, c] = A[row, col]
    '''
    A_c = coo_matrix((Ac_v,(Ac_i,Ac_j)))
    return A_c,F_c,Nodes_wout_IC
