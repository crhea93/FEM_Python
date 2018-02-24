import numpy as np


def Apply_EBC(K,F,Fint,DK_DU,mesh,BC_nodes,BC_vals):
    #first we need to see which nodes are not in the boundary!
    nodes = []
    for i in range(0,2*len(mesh)):
        nodes.append(i)
    Nodes_wout_IC = []
    print("len(mesh): ",len(mesh))
    for i in range(len(nodes)):
        if (nodes[i] in BC_nodes):
            pass
        else:
            Nodes_wout_IC.append(i)
    #print(Nodes_wout_IC)
    print("len(K): ",len(K))
    print("len(Nodes_wout_IC): ",len(Nodes_wout_IC))
    #now to fix F and K matrices
    K_corrected = np.empty((len(Nodes_wout_IC), len(Nodes_wout_IC)))
    dK_corrected = np.empty((len(Nodes_wout_IC), len(Nodes_wout_IC)))
    F_corrected = np.empty((len(Nodes_wout_IC),1))
    for i in range(len(Nodes_wout_IC)):
        current_node_without = Nodes_wout_IC[i]
        for j in range(len(BC_nodes)):
            #print(len(BC_nodes))
            #current_node_with = BC_nodes[j]-1
            current_node_with = BC_nodes[j]
            #print(current_node_with)
            #print(current_node_with,current_node_without)
            #now take away BC_val at node with bc from affected nodes
            if j==0: #if the first one affected
                F_corrected[i] = F[current_node_without] -  BC_vals[j]*K[current_node_without,current_node_with]
            else: #all others based on what we already have for F_corrected just calculated
                F_corrected[i] = F_corrected[i] -  BC_vals[j]*K[current_node_without,current_node_with]
        #end j
    #end i
    Fint_corrected = np.empty((len(Nodes_wout_IC),1))
    counter = 0
    for i in range(0,len(nodes)):
        if i in BC_nodes:
            pass
        else:
            Fint_corrected[counter] = Fint[i]
            counter += 1

    for k in range(len(Nodes_wout_IC)):
        row = Nodes_wout_IC[k]
        for c in range(len(Nodes_wout_IC)):
            col = Nodes_wout_IC[c]
            K_corrected[k, c] = K[row, col]

    for k in range(len(Nodes_wout_IC)):
        row = Nodes_wout_IC[k]
        for c in range(len(Nodes_wout_IC)):
            col = Nodes_wout_IC[c]
            dK_corrected[k, c] = DK_DU[row, col]

    return K_corrected, F_corrected, Fint_corrected, dK_corrected
