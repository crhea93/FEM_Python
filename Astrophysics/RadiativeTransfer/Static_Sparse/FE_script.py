##This program will implement the finite element method for a simple 2D problem in elastostatics
import numpy as np
import numpy.linalg as npl
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
#SuperMatrixU_i = np.array([[]])
#SuperMatrixU_v = np.array([[]])
SuperMatrixF_i = np.array([[]])
SuperMatrixF_v = np.array([[]])
print("Starting loop through Ordinates")
for m in range(M):
    print("We are on ordinate number "+str(m+1))
    T,K,S,F_Source = assembleTKandF(NodalCoord,AngularCoords[m,:], Connectivity,Coefficients, source, El_type, Upwinded)
    A = T+K+S
    F = F_Source
    print("We have assembled the local matrix A and vector F")
    A_corrected,F_Corrected,NodalIDs_wout_EBC = Apply_EBC(A,F,NodalCoord,EssentialBCs,EssentialBCsVals,dictEBC)
    print("We are now placing A_local and F_local into the Super Matrices A and F, resp.")
    for j_local in range(N_correct):
        global_dof1 = m*N_correct+j_local
        if F_Corrected[j_local] != 0:
            SuperMatrixF_i = np.append(SuperMatrixF_i,global_dof1)
            SuperMatrixF_v = np.append(SuperMatrixF_v,F_Corrected[j_local])
        #SuperMatrixF[global_dof1,0] += F_Corrected[j_local]
        for k_local in range(N_correct):
            global_dof2 = m*N_correct+k_local
            if A_corrected[j_local,k_local] != 0:
                SuperMatrixA_i = np.append(SuperMatrixA_i,global_dof1)
                SuperMatrixA_j = np.append(SuperMatrixA_j,global_dof2)
                SuperMatrixA_v = np.append(SuperMatrixA_v,A_corrected[j_local,k_local])
            #SuperMatrixA[global_dof1,global_dof2] += A_corrected[j_local,k_local]
    print("------------------------------------------------------------------------------")
#---------------------------Solve----------------------------------------------#
print("We are solving")
A_csc = csc_matrix((SuperMatrixA_v, (SuperMatrixA_i, SuperMatrixA_j)))
SuperMatrixF_j = np.zeros((len(SuperMatrixF_i)))
F_csc = csc_matrix((SuperMatrixF_v, (SuperMatrixF_i, SuperMatrixF_j)))
F = np.zeros((A_csc.shape[0],1))
count_f = 0
for F_vals in range(len(F)):
    if (F_vals in SuperMatrixF_i):
        F[F_vals,0] = SuperMatrixF_v[count_f]
        count_f += 1
if Solver_type == 'gmres':
    SuperMatrixU = ssl.gmres(A_csc,F)[0]
if Solver_type == 'SuperLU':
    SuperMatrixU_LU = ssl.splu(A_csc)
    SuperMatrixU = SuperMatrixU_LU.solve(F)
else:
    print("Invalid Solver Type")
    print("Default to gmres")
    SuperMatrixU = ssl.gmres(A_csc,F)[0]
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
