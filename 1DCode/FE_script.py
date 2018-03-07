##This program will implement the finite element method for a simple 2D problem in elastostatics
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse.linalg as ssl
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from AssembleMatrices import assembleTandF
from applyEBC import Apply_EBC

#-----------------------------INPUTS-------------------------------------------#
#--------Upwinding Boolean---------------------#
Upwinded = True
#--------Filenames-----------------------------#
filenamvtk = 'test'
#--------Parameters----------------------------#
start_coord = 0
end_coord = 4
N_nodes = 17
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
    Connectivity[i,0] = int(i)
    Connectivity[i,1] = int(i+1)
#--------------------------Assemble Boundaries---------------------------------#
EssentialBCs = [0]
EssentialBCValues = [0.0]
#---------------------------Discretize in Angular Domain-----------------------#
U = np.zeros((N_nodes,1))
T,F_Source = assembleTandF(x_coords, Connectivity,Coefficients, source, Upwinded)
F = F_Source
print("We have assembled the local matrix A and vector F")
A_corrected,F_Corrected = Apply_EBC(T,F,x_coords,EssentialBCs,EssentialBCValues)
#---------------------------Solve----------------------------------------------#
print("We are solving")
U = npl.solve(A_corrected,F_Corrected)
print(U)
#---------------------------Put EBCs Back--------------------------------------#
U_Final = np.zeros((N_nodes,1))
U_Final[0,0] = EssentialBCValues[0]
count = 1
for i in range(N_nodes-len(EssentialBCs)):
    U_Final[count,0] = U[i]
    count += 1
#---------------------------Plot-----------------------------------------------#
approx, = plt.plot(x_coords,U_Final[:,0], marker = "*", label='Approximate Solution')

def analyticSolution(x):
    if x <= 1.5:
        return (x-x**2/2)/a
    elif x<=2.0:
        return (-2*x+x**2/2+2.25)/a
    else:
        return .25/a
x_coords_2 = np.linspace(start_coord,end_coord,N_nodes*100)
AS = np.zeros((len(x_coords_2),1))
for i in range(len(x_coords_2)):
    x_c = x_coords_2[i]
    AS[i,0] = analyticSolution(x_c)

analytic, = plt.plot(x_coords_2,AS[:,0],"--", label='Analytic Solution')
plt.legend(handles=[approx,analytic])
plt.show()

print("Complete")
