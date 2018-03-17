'''
Input file for RTE
'''
from meshes import *
import numpy as np
#-----------------------------INPUTS-------------------------------------------#
El_type = 'H8'
Int_over_ord = True
#--------Constant Values----------------------#
r_c = 0.9
r_s = 0.125
tau = 1.0
x_0 = (10*tau)/(np.arctan(10*r_c))
#--------Coefficient Functions----------------#
def kappa_func(x,y,z):
    return 0
def kappa_func_derx(x,y,z):
    return 0
def kappa_func_dery(x,y,z):
    return 0
def sigma_func(x,y,z,r_c,x_0):
    r = np.sqrt(x**2+y**2+z**2)
    if r <= r_c:
        return x_0/(1+100*r**2)
    else:
        xrc = x_0/(1+100*r_c**2)
        return xrc/100
#--------Source Function-----------------------#
def source(x,y,z,r_s):
    r = np.sqrt(x**2+y**2+z**2)
    if r < r_s:
        return 1
    else:
        return 0
#--------Scatter Type--------------------------#
g = 0 #isotropic scattering
#--------Specify Boundaries and Values---------#
Coefficients = [kappa_func,kappa_func_derx,kappa_func_dery,sigma_func,r_c,r_s,x_0]
#Options include Dirichlet and Searchlight
BCvalsType = 'Dirichlet'
valueBC = 0
#--------Define Direction of Searchlight-------#
AngularCoords = np.matrix([[np.sqrt(2)/2,np.sqrt(2)/2,1]]) # Searchlight
#--------Upwinding Boolean---------------------#
Upwinded = True
#--------Filenames-----------------------------#
meshPhysical = 'mesh/cube_mini.e'
meshAngular = 'mesh/sphere_mini.e'
filenamvtk = 'vtkFiles/scatteringHalo_mini'
#--------Solver Type---------------------------#
Solver_type = "gmres"
#--------Mesh Read in and BC-------------------#
def mesh_read(meshPhysical):
    [NodalCoord,Connectivity,front,back,top,bottom,left,right] = getexodusmesh3DPhysical(meshPhysical)
    EssentialBCs = list(set().union(front,bottom,left))
    return NodalCoord,Connectivity,EssentialBCs
#------------------------------------------------------------------------------#
