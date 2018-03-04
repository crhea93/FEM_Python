import numpy as np
from scipy.sparse import *


def Apply_EBC(A,F,NodesCoord,BC_nodes,BC_vals,dictEBC):
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
            '''if j==0: #if the first one affected
                F_corrected[i] = F[current_node_without] - BC_vals[j]*A[current_node_without,current_node_with]
            else: #all others based on what we already have for F_corrected just calculated
                F_corrected[i] -=  BC_vals[j]*A[current_node_without,current_node_with]'''

    #And now T, which is really simple
    for k in range(len(Nodes_wout_IC)):
        row = Nodes_wout_IC[k]
        for c in range(len(Nodes_wout_IC)):
            col = Nodes_wout_IC[c]
            A_corrected[k, c] = A[row, col]
    return A_corrected,F_corrected,Nodes_wout_IC
