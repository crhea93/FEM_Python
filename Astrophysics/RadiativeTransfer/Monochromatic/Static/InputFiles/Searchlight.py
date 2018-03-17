'''
Input file for RTE
'''
#-----------------------------INPUTS-------------------------------------------#
El_type = 'Q4'
Int_over_ord = False
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
#--------Scatter Type--------------------------#
g = 0 #isotropic scattering
#--------Specify Boundaries and Values---------#
Coefficients = [kappa_func,kappa_func_derx,kappa_func_dery,sigma_func]
#Options include Dirichlet and Searchlight
BCvalsType = '2D set'
valueBC = 1.0
#--------Define Direction of Searchlight-------#
AngularCoords = np.matrix([[np.sqrt(2)/2,np.sqrt(2)/2,1]]) # Searchlight
#--------Upwinding Boolean---------------------#
Upwinded = True
#--------Filenames-----------------------------#
meshPhysical = 'mesh/square_ref2.e'
meshAngular = 'mesh/sphere.e'
filenamvtk = 'vtkFiles/Searchlight_ref2'
#--------Solver Type---------------------------#
Solver_type = "gmres"
#--------Mesh Read in and BC-------------------#
def mesh_read(meshPhysical):
    [NodalCoord,Connectivity,left,bottom,right,top] = getexodusmesh2D(meshPhysical)
    in_bottom_but_not_left = set(list(bottom)) - set(list(left))
    EssentialBCs = list(left) + list(in_bottom_but_not_left)
return NodalCoord,Connectivity,EssentialBCs
#------------------------------------------------------------------------------#
