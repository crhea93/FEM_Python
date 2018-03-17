import numpy as np
from scipy.sparse import *


def Apply_EBC(A_i,A_j,T_v,K_v,S_v,F,NodesCoord,BC_nodes,BC_vals,dictEBC):
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
    Tc_v = np.array([[]])
    Kc_v = np.array([[]])
    Sc_v = np.array([[]])
    F_c = np.zeros(num_wout, dtype = float)
    A = csc_matrix((T_v+K_v+S_v,(A_i,A_j)))
    A = A.todok()
    #Afull = A.todense()
    #A_cdense = np.zeros((num_wout,num_wout))
    #A_dok = dok_matrix((num_wout,num_wout), dtype = float)



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


    #for k in range(len(Nodes_wout_IC)):
    #    row = Nodes_wout_IC[k]
    #    for c in range(len(Nodes_wout_IC)):
    #        col = Nodes_wout_IC[c]
    #        A_cdense[k, c] = Afull[row, col] #uncomment to make it work :(

    in_both = set(A_i).intersection(Nodes_wout_IC)
    ind_dict = dict((k,i) for i,k in enumerate(A_i))
    inter = set(ind_dict).intersection(Nodes_wout_IC) #lit of global dof (key for ind_dict)
    glob_to_red = dict()
    index_count = 0
    for key in inter:
        glob_to_red[key] = int(index_count)
        index_count += 1
    for walker in range(len(A_i)):
        i_ind = int(A_i[walker])
        j_ind = int(A_j[walker])
        if i_ind in in_both and j_ind in in_both:
            Ac_i = np.append(Ac_i,glob_to_red[i_ind])
            Ac_j = np.append(Ac_j,glob_to_red[j_ind])
            Tc_v = np.append(Tc_v,T_v[walker])
            Kc_v = np.append(Kc_v,K_v[walker])
            Sc_v = np.append(Sc_v,S_v[walker])

    #A_c = coo_matrix((Ac_v,(Ac_i,Ac_j)))
    return Ac_i,Ac_j,Tc_v,Kc_v,Sc_v,F_c,Nodes_wout_IC,glob_to_red
