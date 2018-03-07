##This program will implement the finite element method for a simple 2D problem in elastostatics
import numpy as np
import numpy.linalg as npl
from numba import jit
import scipy.linalg as spl
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
from meshes import getexodusmesh2D,getexodusmesh3D
from AssembleMatrices import assembleTKandF
from applyEBC import Apply_EBC
from BCValues import getBCValues,getBCs
import time
from Vtkwriter import vtkwritefield
from scipy.sparse import *
import time

from InputFiles.Searchlight import *



#--------------------------Start of Actual FEM Program-------------------------#
start = time.time()
#--------------------------Read in meshes--------------------------------------#
[NodalCoord,Connectivity,left,bottom,right,top] = getexodusmesh2D(meshPhysical)
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
if Int_over_ord == True:
    [AngularCoords,AngularConnectivity] = getexodusmesh3D(meshAngular)
    M = len(AngularCoords)
else:
    M = 1
    AngularCoords = np.matrix([[np.sqrt(2)/2,np.sqrt(2)/2,1]]) # Searchlight
N_correct = N - len(EssentialBCs) #Number of DoF excluding EBCs
Corrected_size = N*M-M*len(EssentialBCs) # Must subtract off number of EBC to
#get corrected sizes so we can apply EBC within each Ordinate Step
#Set up IJV (COO) Matrices
SuperMatrixA_i = np.array([[]])
SuperMatrixA_j = np.array([[]])
SuperMatrixA_v = np.array([[]])
SuperMatrixF_i = np.array([[]])
SuperMatrixF_v = np.array([[]])
print("Starting loop through Ordinates")
startTime = time.time()
for m in range(M):
    print("We are on ordinate number "+str(m+1))
    Ap_i,Ap_j,Ap_v,F = assembleTKandF(NodalCoord,AngularCoords[m,:], Connectivity,Coefficients, source, El_type, Upwinded)
    print("We have assembled the local matrix A and vector F")
    #-----------Create Dictionary for sparse nodes-----------------------------#
    A_corrected,F_corrected,NodalIDs_wout_EBC = Apply_EBC(Ap_i,Ap_j,Ap_v,F,NodalCoord,EssentialBCs,EssentialBCsVals,dictEBC)
    '''
    print("We are now placing A_local and F_local into the Super Matrices A and F, resp.")
    for j_local in range(N_correct):
        global_dof1 = m*N_correct+j_local
        if F_Corrected[j_local] != 0:
            SuperMatrixF_i = np.append(SuperMatrixF_i,global_dof1)
            SuperMatrixF_v = np.append(SuperMatrixF_v,F_Corrected[j_local])
        for k_local in range(N_correct):
            global_dof2 = m*N_correct+k_local
            if A_corrected[j_local,k_local] != 0:
                SuperMatrixA_i = np.append(SuperMatrixA_i,global_dof1)
                SuperMatrixA_j = np.append(SuperMatrixA_j,global_dof2)
                SuperMatrixA_v = np.append(SuperMatrixA_v,A_corrected[j_local,k_local])'''
    print("------------------------------------------------------------------------------")
end_time_ord = time.time()

#-------------------------- Get F vector the correct size----------------------#

#---------------------------Solve----------------------------------------------#
if Solver_type == "gmres":
    print("Using GMRES")
    SuperMatrixU = ssl.gmres(A_corrected,F_corrected)[0]
if Solver_type == "SuperLU":
    print("Using SuperLU")
    SuperMatrixU_LU = ssl.splu(A_corrected)
    SuperMatrixU = SuperMatrixU_LU.solve(F)

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


print("Total time is "+str(round(end_time_ord-startTime,2))+" seconds.")


print("Complete")
