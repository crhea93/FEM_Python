'''
Boundary values for FEM
'''
import numpy as np
def getBCValues(NodalCoord,EssentialBCs,BCvalsType,value):
    BCValues = np.zeros((len(EssentialBCs),1))
    if BCvalsType == 'Searchlight':
        bc_count = 0
        for bcs_i in EssentialBCs:
            current_x = NodalCoord[bcs_i-1,0]
            current_y = NodalCoord[bcs_i-1,1]
            BCValues[bc_count] = SpotLightFunc(current_x,current_y,value)
            bc_count += 1

    if BCvalsType == 'Dirichlet':
        for bcs_i in range(len(BCValues)):
            BCValues[bcs_i] = value
    return BCValues

def getBCs(NodalCoord,value):
    BCNodes = []
    BCValues = []
    for bcs_i in range(len(NodalCoord)):
        current_x = NodalCoord[bcs_i,0]
        current_y = NodalCoord[bcs_i,1]
        splfuncval = SpotLightFunc(current_x,current_y,value)
        if splfuncval != 0:
            BCNodes.append(bcs_i+1)
            BCValues.append(splfuncval)
    return BCNodes,BCValues

def SpotLightFunc(x,y,val):
    if (x==-1.0) and (y>-0.9 and y<-0.50):
        return val
    else:
        return 0
