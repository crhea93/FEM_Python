#File which handles read in of mesh
'''
Need to output the Nodal Coordinates, Connectivity Array, and Essential BC nodes
'''
import numpy as np
from netCDF4 import Dataset

def getMeshSimple():
    Xcoord = np.array((0,10,20,2,5,17))
    Ycoord = np.array((0,2,0,6,8,6))
    NodalCoords = np.array([Xcoord.transpose(),Ycoord.transpose()])
    #print(NodalCoords[:,1])
    Connectivity = np.array([(1,2,5,4),(2,3,6,5)])
    EssentialBCs = np.array((1,2,4,6))
    return [NodalCoords.transpose(),Connectivity,EssentialBCs]


def getexodusmesh():
    exodus_file = 'mesh/bar.e'
    element_type = 'Q4' #or 'T3'
    nc = Dataset(exodus_file)
    #print(nc)
    x = nc.variables['coord'][0]
    y = nc.variables['coord'][1]
   # z = nc.variables['coord'][2]
    connect = nc.variables['connect1'][:]
    left = nc.variables['node_ns1'][:]
    bottom = nc.variables['node_ns2'][:]
    right = nc.variables['node_ns3'][:]
    top = nc.variables['node_ns4'][:]

    #print(connect)
   # Xcoord = np.array(x)
   # Ycoord = np.array(y)
   # Connectivity = np.array(connect)
    #print(nc.variables['ns_status'])
    #print(len(x))
    nodes = np.zeros((len(x),2))
    for i in range(0,len(x)):
        nodes[i,0] = x[i]
        nodes[i,1] = y[i]
    return [nodes,connect,left,bottom,right,top]
    

def readTxt(mesh,connect,sides):
    # This opens a handle to the file, in 'r' read mode
    file_handle = open(mesh, 'r')
    # Read in all the lines of your file into a list of lines
    lines_list = file_handle.readlines()
    # Extract dimensions from first line. Cast values to integers from strings.
    x, y = (float(val)-1 for val in lines_list[0].split())
    # Do a double-nested list comprehension to get the rest of the data into your matrix
    coordinates = [[float(val) for val in line.split()] for line in lines_list[:]]
    #in the form of [x,y]
    #Lets grab the connectivity matrix! form of [n1,n2,n3,n4] for each element! So row - element, col - node
    # This opens a handle to the file, in 'r' read mode
    file_handle = open(connect, 'r')
    # Read in all the lines of your file into a list of lines
    lines_list = file_handle.readlines()
    # Extract dimensions from first line. Cast values to integers from strings.
    # N1,N2,N3,N4 = (int(val)-1 for val in lines_list[0].split())
    # Do a double-nested list comprehension to get the rest of the data into your matrix
    connectivity_matrix = [[int(val)-1 for val in line.split()] for line in lines_list[:]]
    #to access row, con_mat[i] and element con_mat[i][j]
    
    #read sides
    # Lets grab the connectivity matrix! form of [n1,n2,n3,n4] for each element! So row - element, col - node
    # This opens a handle to the file, in 'r' read mode
    file_handle = open(sides, 'r')
    # Read in all the lines of your file into a list of lines
    lines_list = file_handle.readlines()
    # Extract dimensions from first line. Cast values to integers from strings.
    #S1,S2,S3,S4 = (int(val)-1 for val in lines_list[0].split())
    # Do a double-nested list comprehension to get the rest of the data into your matrix
    sides_mat = [[int(val) - 1 for val in line.split()] for line in lines_list[:]]
    top = []; left = []; bottom = []; right = []
    for i in range(len(sides_mat)-1):
        top.append(sides_mat[i][0])
        left.append(sides_mat[i][1])
        bottom.append(sides_mat[i][2])
        right.append(sides_mat[i][3])
    
    
    return coordinates,connectivity_matrix,top,left,bottom,right
