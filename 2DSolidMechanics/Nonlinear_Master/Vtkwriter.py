#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:21:11 2017
VTK Writer
@author: crhea
"""
def vtkwrite(filename,num_nodes_total,num_el,x_coords,y_coords,mesh):
    vtk = open(filename+".vtk",'w')
    vtk.write("# vtk DataFile Version 2.0")
    vtk.write('\n')
    vtk.write("Packing Fraction Fields")
    vtk.write('\n')
    vtk.write("ASCII")
    vtk.write('\n')
    vtk.write("DATASET UNSTRUCTURED_GRID")
    vtk.write('\n')
    vtk.write("POINTS "+str(num_nodes_total)+" double"+'\n')
    for i in range(0,num_nodes_total):
        vtk.write(str(x_coords[i])+" "+str(y_coords[i])+" "+"0.0"+'\n')
    vtk.write('\n')
    vtk.write("CELLS "+str(num_el)+" "+str(5*num_el)+'\n')
    for i in range(0,num_el):
        vtk.write("4 "+str(mesh[i][1]-1)+" "+str(mesh[i][0]-1)+" "+str(mesh[i][3]-1)+" "+str(mesh[i][2]-1)+'\n')
    vtk.write('\n')
    vtk.write("CELL_TYPES "+str(num_el)+'\n')
    for i in range(0,num_el):
        vtk.write("9"+'\n')
    vtk.close()


def vtkwritefield(filename,num_nodes_total,num_el,x_coords,y_coords,mesh,field):
    vtk = open(filename+".vtk",'w')
    vtk.write("# vtk DataFile Version 2.0")
    vtk.write('\n')
    vtk.write("Packing Fraction Fields")
    vtk.write('\n')
    vtk.write("ASCII")
    vtk.write('\n')
    vtk.write("DATASET UNSTRUCTURED_GRID")
    vtk.write('\n')
    vtk.write("POINTS "+str(num_nodes_total)+" double"+'\n')
    for i in range(0,num_nodes_total):
        vtk.write(str(x_coords[i])+" "+str(y_coords[i])+" "+"0.0"+'\n')
    vtk.write('\n')
    vtk.write("CELLS "+str(num_el)+" "+str(5*num_el)+'\n')
    for i in range(0,num_el):
        vtk.write("4 "+str(mesh[i][1]-1)+" "+str(mesh[i][0]-1)+" "+str(mesh[i][3]-1)+" "+str(mesh[i][2]-1)+'\n')
    vtk.write('\n')
    vtk.write("CELL_TYPES "+str(num_el)+'\n')
    for i in range(0,num_el):
        vtk.write("9"+'\n')
    vtk.write("CELL DATA "+str(num_el)+'\n')
    vtk.write("SCALARS cell_ PackingFraction double"+'\n')
    vtk.write("LOOKUP_TABLE default"+'\n')
    for i in range(num_el):
        vtk.write(str(field[0,i])+'\n')
    vtk.close()
   