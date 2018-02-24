##This program will implement the finite element method for a simple 2D problem in elastostatics
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt
from meshes import getMeshSimple,getexodusmesh,readTxt
from AssembleStiffnessMatrix import assembleNL
from getForceFromGravity import getForceFromGravity
from applyEBC import Apply_EBC
from StressStrain import getStrain
import time
from Vtkwriter import vtkwrite

#-------------------------------------------------------------------------#
#inputs
ToL = 1e-5
E = 200e9
nu = 0.3
rho = 7800
h = 1.0
g = 9.8
a = E/(1-nu**2)
#-------------------------------------------------------------------------#
# initialize the D vector based off classical definition
def Dns(epsilon):
    return a * np.matrix([[1, nu, 0] ,[nu ,1 ,0] ,[0, 0, (1-nu)/2]])
def D(eps,pt):
    return Dns(eps)

def D_prime(sig):
    return a*np.matrix([[0,0,0],[0,0,0],[0,0,0]])

pseudo_time_max = 1 + 1 #second value is the number of steps
endTime = 10 #end of time
time_ = np.linspace(0,endTime,pseudo_time_max)

start = time.time()
#---------------------Read in mesh-------------------------
[NodalCoord, Connectivity,left,bottom,right,top] = getexodusmesh()
left = list(left); bottom = list(bottom); right = list(right); top = list(top)

#---------------------Assemble Boundaries -----------------
EssentialBcs = list(left) + list(right)
EssentialBcsx = [2*(x-1) for x in EssentialBcs]
EssentialBcsy = [2*(x-1)+1 for x in EssentialBcs]
EssentialBcs = EssentialBcsx+EssentialBcsy
EssentialBcsVals = np.zeros(len(EssentialBcs))
#initialize stress and strain
epsilon = np.zeros((3,len(Connectivity)*4))
sigma = np.zeros((3,len(Connectivity)*4))
print("Mesh has been read in and we are about to begin the Newton-Raphson solve")
#only do NR solve with corrected
print("len(NodalCoord):",len(NodalCoord))

total_disp = np.zeros((2 * len(NodalCoord),1))

disp = np.zeros((2 * len(NodalCoord)-len(EssentialBcs),1))
pt = 1 #initialize load  step number
while pt < (pseudo_time_max):
    print("We are on load step number: ",pt)
    Fg = getForceFromGravity(NodalCoord, Connectivity, time_[pt]*rho, g, h) #calculate force from gravity
    F = Fg
    rnorm = 100 #initialize norm to force solve
    nriter = 1 #newton raphson iteration count
    deltad = np.zeros((2*len(NodalCoord),1))#incremental steps
    while (rnorm>ToL):
        print("We are on iteration ",nriter," of the NR solve")
        K,A,Fsig,dk_du= assembleNL(NodalCoord, Connectivity, D, D_prime,h,epsilon,pt,sigma)#this handles the entirety of assembling the local stiffness matrices based of the equations
        K_corrected, F_corrected, F_sig,DK_DU = Apply_EBC(K,F,Fsig,dk_du,NodalCoord,EssentialBcs,EssentialBcsVals)#Here we simply are applying the essential boundary conditions
        F_int = np.dot(K_corrected,disp) #Solving for F_internal using the classical manner
        Ktan = K_corrected+np.dot(DK_DU,disp) #Classic equation for consistent tangent
        Residual = F_corrected-(F_int)
        print("The norm of the difference between the elementwise calculated Fint and final Fint is: ", spl.norm(F_int-F_sig))
        if (nriter==1): #set init values for reference
            deltad0 = spl.solve(Ktan,Residual)
            Residual0 = Residual
            deltad = deltad0
        else:
            deltad = spl.solve(Ktan,Residual) #solve K_tan*delta_d=R
        disp += deltad
        count = 0
        bc_count = 0
	#now to put the "boundaries" back
        for i in range(0, len(total_disp)):
            if i  in EssentialBcs:
                total_disp[i] = EssentialBcsVals[bc_count]
                bc_count += 1
            else:
                total_disp[i] = disp[count]
                count += 1
	#update stress and strains
        epsilon,sigma = getStrain(NodalCoord, Connectivity, total_disp, D, pt,A)
        nriter += 1
	#calculate norm
        rnorm = np.abs(np.dot(deltad.transpose(),Residual)/np.dot(deltad0.transpose(),Residual0))
        print("The norm of the residual is: ",rnorm[0][0])


    pt += 1 #next pseudo time step
    print()

#now to put everything back together and plot
scale  = 1
x_coords = np.full(len(NodalCoord),0,dtype='float64')
y_coords = np.full(len(NodalCoord),0,dtype='float64')

count = 0
for i in range(len(x_coords)):
    dof1 = 2*(i)
    dof2 = 2*(i)+1
    x_coords[i] = NodalCoord[i][0]+total_disp[dof1]*scale
    y_coords[i] = NodalCoord[i][1]+total_disp[dof2]*scale
    count += 1



x_inits = np.full(len(NodalCoord),0,dtype='float64')
y_inits = np.full(len(NodalCoord),0,dtype='float64')

for i in range(len(x_coords)):
    x_inits[i] = NodalCoord[i][0]
    y_inits[i] = NodalCoord[i][1]

#save plots as vtk files


vtkwrite("output/undeformed",len(NodalCoord),len(Connectivity),x_inits,y_inits,Connectivity)
vtkwrite("output/deformed",len(NodalCoord),len(Connectivity),x_coords,y_coords,Connectivity)
end = time.time()
print("Total time is: ",str(end-start))
