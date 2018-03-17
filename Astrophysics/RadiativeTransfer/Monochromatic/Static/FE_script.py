##This program will implement the finite element method for a simple 2D problem in elastostatics
import numpy as np
#import numpy.linalg as npl
from numba import jit
#import scipy.linalg as spl
import scipy.sparse.linalg as ssl
from scipy.sparse import *
import matplotlib.pyplot as plt
from meshes import getexodusmesh2D,getexodusmesh3D
from AssembleMatrices import assemble
from Local_Matrices import CalcNorm2D, CalcNorm3D
from applyEBC import Apply_EBC
from BCValues import getBCValues,getBCs
from Vtkwriter import vtkwritefield
from Redistribution import RedistributionFunc

import time

from InputFiles.ScatteringHalo import *



#--------------------------Start of Actual FEM Program-------------------------#
start = time.time()
#--------------------------Read in meshes--------------------------------------#
#[NodalCoord,Connectivity,left,bottom,right,top,front,back] = getexodusmesh3DPhysical(meshPhysical)
NodalCoord,Connectivity,EssentialBCs = mesh_read(meshPhysical)
print("Mesh has been read in")
#--------------------------Assemble Boundaries---------------------------------#
#-------Physical Domain-------------------------#
print("Setting up Essential Boundary Conditions")
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
N_correct = N - len(EssentialBCs) #Number of DoF excluding EBCs
Corrected_size = N*M-M*len(EssentialBCs) # Must subtract off number of EBC to
#get corrected sizes so we can apply EBC within each Ordinate Step
#Set up IJV (COO) Matrices
SuperMatrixA_i = np.array([[]])
SuperMatrixA_j = np.array([[]])
SuperMatrixA_v = np.array([[]])
SuperMatrixF = np.array([[]])
print("Starting loop through Ordinates")
startTime = time.time()
for m in range(M):
    print("We are on ordinate number "+str(m+1))
    print("We have assembled the physical matrix A and vector F")
    if El_type == 'Q4':
        n_vec_m = CalcNorm2D(AngularCoords[m,:])
    if El_type == 'H8':
        n_vec_m = CalcNorm3D(AngularCoords[m,:])
    Ap_i,Ap_j,Tp_v,Kp_v,Sp_v,F = assemble(NodalCoord,n_vec_m, Connectivity,Coefficients, source, El_type, Upwinded)
    print("We are now applying the Essential Boundary Conditions")
    #-----------Create Dictionary for sparse nodes-----------------------------#
    Ac_i,Ac_j,Tc_v,Kc_v,Sc_v,F_corrected,NodalIDs_wout_EBC,G_Red_map = Apply_EBC(Ap_i,Ap_j,Tp_v,Kp_v,Sp_v,F,NodalCoord,EssentialBCs,EssentialBCsVals,dictEBC)
    print("We are assembling the global matrix")
    for count_global in range(len(Ac_i)):
        i_ind = int(Ac_i[count_global])+m*N_correct
        j_ind = int(Ac_j[count_global])+m*N_correct
        w = RedistributionFunc(g,n_vec_m[0])/M
        SuperMatrixA_i = np.append(SuperMatrixA_i,i_ind)
        SuperMatrixA_j = np.append(SuperMatrixA_j,j_ind)
        SuperMatrixA_v = np.append(SuperMatrixA_v,Tc_v[count_global] + Kc_v[count_global]+w*Sc_v[count_global])
        for m2 in range(M):
            if m2 != m:
                if El_type == 'Q4':
                    n_vec_m2 = CalcNorm2D(AngularCoords[m2,:])
                if El_type == 'H8':
                    n_vec_m2 = CalcNorm3D(AngularCoords[m2,:])
                w = RedistributionFunc(g,n_vec_m2[0])/M
                j_s = j_ind + m2*N_correct
                SuperMatrixA_i = np.append(SuperMatrixA_i,i_ind)
                SuperMatrixA_j = np.append(SuperMatrixA_j,j_s)
                SuperMatrixA_v = np.append(SuperMatrixA_v,w*Sc_v[count_global])

    for i_f in range(len(F_corrected)):
        id_global = i_f + m*N_correct
        SuperMatrixF = np.append(SuperMatrixF,F_corrected[i_f])

    print("------------------------------------------------------------------------------")
end_time_ord = time.time()
SuperMatrixA = coo_matrix((SuperMatrixA_v,(SuperMatrixA_i,SuperMatrixA_j)))
SuperMatrixF = F_corrected
#---------------------------Solve----------------------------------------------#
if Solver_type == "gmres":
    print("Using GMRES")
    SuperMatrixU = ssl.gmres(SuperMatrixA,SuperMatrixF)[0]
if Solver_type == "SuperLU":
    print("Using SuperLU")
    SuperMatrixU_LU = ssl.splu(SuperMatrixA)
    SuperMatrixU = SuperMatrixU_LU.solve(SuperMatrixF.todense())
#---------------------------Carry Out Angular Integration----------------------#
print("Now carrying out our angular integration")
U_angular = np.zeros((N_correct,1))
for ordinate in range(M):
    for dof in range(N_correct):
        U_angular[dof] += SuperMatrixU[ordinate*N_correct+dof]
U_angular = U_angular/M
#----------------------------Create Dictionary of solutions--------------------#
dictsol = {}
for key,val in G_Red_map.items():
    dictsol[key] = U_angular[val]
#---------------------------Put EBCs Back--------------------------------------#
U_Final = np.zeros((N,1))
for NodeID,Nodevalue in dictEBC.items():
    U_Final[NodeID-1] += Nodevalue
for NodeID,Nodevalue in dictsol.items():
    U_Final[NodeID] += Nodevalue
#---------------------------Plot-----------------------------------------------#
x = np.array(NodalCoord[:,0])
y = np.array(NodalCoord[:,1])
vtkwritefield(filenamvtk,N,len(Connectivity),x,y,Connectivity,U_Final)

if filenamvtk == 'vtkFiles/scatteringHalo':
    x = np.array(NodalCoord[:,0])
    y = np.array(NodalCoord[:,1])
    z = np.array(NodalCoord[:,2])
    r = np.sqrt(x**2+y**2+z**2)
    plt.plot(r,U_Final)

print("Total time is "+str(round(end_time_ord-startTime,2))+" seconds.")


print("Complete")
