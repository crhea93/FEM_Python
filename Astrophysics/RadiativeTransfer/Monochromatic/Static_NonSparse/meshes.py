#File which handles read in of mesh
'''
Need to output the Nodal Coordinates, Connectivity Array, and Essential BC nodes
'''
import numpy as np
from netCDF4 import Dataset


def getexodusmesh2D(filename):
    exodus_file = filename
    nc = Dataset(exodus_file)
    x = nc.variables['coord'][0]
    y = nc.variables['coord'][1]
    connect = nc.variables['connect1'][:]
    left = nc.variables['node_ns1'][:]
    bottom = nc.variables['node_ns2'][:]
    right = nc.variables['node_ns3'][:]
    top = nc.variables['node_ns4'][:]
    nodes = np.zeros((len(x),2))
    for i in range(0,len(x)):
        nodes[i,0] = x[i]
        nodes[i,1] = y[i]
    return [nodes,connect,left,bottom,right,top]


def getexodusmesh3D(filename):
    exodus_file = filename
    nc = Dataset(exodus_file)
    x = nc.variables['coord'][0]
    y = nc.variables['coord'][1]
    z = nc.variables['coord'][2]
    connect = nc.variables['connect1'][:]
    nodes = np.zeros((len(x),3))
    for i in range(0,len(x)):
        nodes[i,0] = x[i]
        nodes[i,1] = y[i]
        nodes[i,2] = z[i]
    return [nodes,connect]
