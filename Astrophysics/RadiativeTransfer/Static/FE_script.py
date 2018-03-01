##This program will implement the finite element method for a simple 2D problem in elastostatics
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
from meshes import getexodusmesh2D,getexodusmesh3D
from AssembleMatrices import assembleTandF
from applyEBC import Apply_EBC
from BCValues import getBCValues,getBCs
import time
from Vtkwriter import vtkwritefield

#-----------------------------INPUTS-------------------------------------------#
El_type = 'Q4'
#--------Coefficient Functions----------------#
def kappa_func(x,y):
    return 0
def kappa_func_derx(x,y):
    return 0
def kappa_func_dery(x,y):
    return 0
def sigma_func(x,y):
    return 0
#--------Source Function-----------------------#
def source(x,y):
    return 0
#--------Specify Boundaries and Values---------#
Coefficients = [kappa_func,kappa_func_derx,kappa_func_dery,sigma_func]
#Options include Dirichlet and Searchlight
BCvalsType = 'Searchlight'
valueBC = 1.0
#--------Upwinding Boolean---------------------#
Upwinded = True
#--------Filenames-----------------------------#
meshPhysical = 'mesh/square.e'
meshAngular = 'mesh/sphere.e'
filenamvtk = 'test'
#------------------------------------------------------------------------------#














#--------------------------Start of Actual FEM Program-------------------------#
start = time.time()
#--------------------------Read in meshes--------------------------------------#
[NodalCoord,Connectivity,left,bottom,right,top] = getexodusmesh2D(meshPhysical)
[AngularCoords,AngularConnectivity] = getexodusmesh3D(meshAngular)
print("Mesh has been read in")
#--------------------------Assemble Boundaries---------------------------------#
#-------Physical Domain-------------------------#
in_bottom_but_not_left = set(list(bottom)) - set(list(left))
EssentialBCs = list(left) + list(in_bottom_but_not_left)
EssentialBCsVals = getBCValues(NodalCoord,EssentialBCs,BCvalsType,valueBC)
dictEBC = {}
count = 0
for i in EssentialBCs:
        dictEBC[i] = EssentialBCsVals[count]
        count += 1
#---------------------------Discretize in Angular Domain-----------------------#
N = len(NodalCoord) # Number of degrees of freedom including EBCs
M = len(AngularCoords) # Number of Ordinates
N_correct = N - len(EssentialBCs) #Number of DoF excluding EBCs
Corrected_size = N*M-M*len(EssentialBCs) # Must subtract off number of EBC to
#get corrected sizes so we can apply EBC within each Ordinate Step
SuperMatrixA = np.zeros((Corrected_size,Corrected_size))
SuperMatrixU = np.zeros((Corrected_size,1))
SuperMatrixF = np.zeros((Corrected_size,1))
print("Starting loop through Ordinates")
for m in range(M):
    print("We are on ordinate number "+str(m+1))
    T,F_Source = assembleTandF(NodalCoord,AngularCoords[m,:], Connectivity,Coefficients, source, El_type, Upwinded)
    A = T
    F = F_Source
    print("We have assembled the local matrix A and vector F")
    A_corrected,F_Corrected,NodalIDs_wout_EBC = Apply_EBC(A,F,NodalCoord,EssentialBCs,EssentialBCsVals,dictEBC)
    print("We are now placing A_local and F_local into the Super Matrices A and F, resp.")
    for j_local in range(len(A_corrected)):
        global_dof1 = m*N_correct+j_local
        SuperMatrixF[global_dof1,0] += F_Corrected[j_local]
        for k_local in range(len(A_corrected)):
            global_dof2 = m*N_correct+k_local
            SuperMatrixA[global_dof1,global_dof2] += A_corrected[j_local,k_local]
    print("------------------------------------------------------------------------------")
#---------------------------Solve----------------------------------------------#
print("We are solving")
SuperMatrixU = spl.solve(SuperMatrixA,SuperMatrixF)
#SuperMatrixU = ssl.lsqr(SuperMatrixA,SuperMatrixF)
SuperMatrixU = SuperMatrixU[:]
#---------------------------Carry Out Angular Integration----------------------#
print("Now carrying out our angular integration")
U_angular = np.zeros((N_correct,1))
for ordinate in range(M):
    for dof in range(N_correct):
        U_angular[dof] += SuperMatrixU[ordinate*N_correct+dof]
U_angular = U_angular/M
#----------------------------Create Dictionary of solutions--------------------#
dictsol = {}
count_sol_dict = 0
for seul in NodalIDs_wout_EBC:
    dictsol[seul] = U_angular[count_sol_dict]
    count_sol_dict += 1
#---------------------------Put EBCs Back--------------------------------------#
U_Final = np.zeros((N,1))
for NodeID,Nodevalue in dictEBC.items():
    U_Final[NodeID-1] = Nodevalue
for NodeID,Nodevalue in dictsol.items():
    U_Final[NodeID] = Nodevalue
#---------------------------Plot-----------------------------------------------#
x = np.array(NodalCoord[:,0])
y = np.array(NodalCoord[:,1])
vtkwritefield(filenamvtk,N,len(Connectivity),x,y,Connectivity,U_Final)

#for i in range(N):
#    print(str(x[i])+" "+str(y[i])+" "+str(U_Final[i]))





print("Complete")
