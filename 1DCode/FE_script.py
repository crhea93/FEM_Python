##This program will implement the finite element method for a simple 2D problem in elastostatics
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
from AssembleMatrices import assembleTandF
from applyEBC import Apply_EBC

#-----------------------------INPUTS-------------------------------------------#
#--------Upwinding Boolean---------------------#
Upwinded = False
#--------Filenames-----------------------------#
filenamvtk = 'test'
#--------Parameters----------------------------#
start_coord = 0
end_coord = 4
N_nodes = 1001
a = 1
k = 0
def source(x):
    if x <= 1.5:
        return 1-x
    elif x<=2.0:
        return -2+x
    else:
        return 0
#------------------------------------------------------------------------------#



#--------------------------Start of Actual FEM Program-------------------------#
Coefficients = [a,k]
#--------------------------Read in meshes--------------------------------------#
x_coords = np.linspace(start_coord,end_coord,N_nodes)
Connectivity = np.zeros((N_nodes-1,2),dtype = int)
for i in range(N_nodes-1):
    Connectivity[i,0] = int(i+1)
    Connectivity[i,1] = int(i+2)
#--------------------------Assemble Boundaries---------------------------------#
EssentialBCs = [0]
EssentialBCValues = [0]
#---------------------------Discretize in Angular Domain-----------------------#
U = np.zeros((N_nodes,1))
T,F_Source = assembleTandF(x_coords, Connectivity,Coefficients, source, Upwinded)
F = F_Source
print("We have assembled the local matrix A and vector F")
A_corrected,F_Corrected = Apply_EBC(T,F,x_coords,EssentialBCs,EssentialBCValues)
#---------------------------Solve----------------------------------------------#
print("We are solving")
U = npl.solve(A_corrected,F_Corrected)

#---------------------------Put EBCs Back--------------------------------------#
U_Final = np.zeros((N_nodes,1))
U_Final[0,0] = EssentialBCValues[0]
for i in range(N_nodes-len(EssentialBCs)):
    count = len(EssentialBCs)+i
    U_Final[count] = U[i]
#---------------------------Plot-----------------------------------------------#
plt.plot(x_coords,U_Final[:,0])

def analyticSolution(x):
    if x <= 1.5:
        return (x-x**2/2)/a
    elif x<=2.0:
        return (-2*x+x**2/2+2.25)/a
    else:
        return .25/a
AS = np.zeros((len(x_coords),1))
for i in range(len(x_coords)):
    x_c = x_coords[i]
    AS[i,0] = analyticSolution(x_c)

plt.plot(x_coords,AS[:,0],"--")
plt.show()

print("Complete")
