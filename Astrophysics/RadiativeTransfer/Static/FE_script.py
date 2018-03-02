##This program will implement the finite element method for a simple 2D problem in elastostatics
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
from krypy.linsys import LinearSystem, Minres
from scipy.sparse.linalg import dsolve
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
if Int_over_ord == True:
    M = len(AngularCoords)
else:
    M = 1
    AngularCoords = np.matrix([[np.sqrt(2)/2,np.sqrt(2)/2,1]]) # Searchlight
N_correct = N - len(EssentialBCs) #Number of DoF excluding EBCs
Corrected_size = N*M-M*len(EssentialBCs) # Must subtract off number of EBC to
print(Corrected_size)
#get corrected sizes so we can apply EBC within each Ordinate Step
#SuperMatrixA = np.zeros((Corrected_size,Corrected_size))
SuperMatrixA = np.zeros((Corrected_size,Corrected_size))
SuperMatrixU = np.zeros((Corrected_size,1))
SuperMatrixF = np.zeros((Corrected_size,1))
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
        SuperMatrixF[global_dof1,0] += F_Corrected[j_local]
        for k_local in range(N_correct):
            global_dof2 = m*N_correct+k_local
            SuperMatrixA[global_dof1,global_dof2] += A_corrected[j_local,k_local]
    print("------------------------------------------------------------------------------")
#---------------------------Solve----------------------------------------------#
print("We are solving")
#SuperMatrixU = spl.solve(SuperMatrixA,SuperMatrixF)
#SuperMatrixU = dsolve.spsolve(SuperMatrixA, SuperMatrixF, use_umfpack=False)
#SuperMatrixU = ssl.lsqr(SuperMatrixA,SuperMatrixF)
#linear_system = LinearSystem(SuperMatrixA, SuperMatrixF, self_adjoint=True)
#solver = Minres(linear_system)
SuperMatrixU = ssl.splu(csc_matrix(SuperMatrixA)).solve(SuperMatrixF)

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
