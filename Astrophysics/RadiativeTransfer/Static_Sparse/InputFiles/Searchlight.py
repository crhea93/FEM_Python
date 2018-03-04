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
#--------Specify Boundaries and Values---------#
Coefficients = [kappa_func,kappa_func_derx,kappa_func_dery,sigma_func]
#Options include Dirichlet and Searchlight
BCvalsType = 'Searchlight'
valueBC = 1.0
#--------Upwinding Boolean---------------------#
Upwinded = True
#--------Filenames-----------------------------#
meshPhysical = 'mesh/square_ref1.e'
meshAngular = 'mesh/sphere.e'
filenamvtk = 'Searchlight_refq'
#--------Solver Type---------------------------#
Solver_type = 'gmres'
#------------------------------------------------------------------------------#
