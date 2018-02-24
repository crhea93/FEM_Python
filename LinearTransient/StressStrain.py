#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:47:44 2017



@author: crhea
"""


from getElementCoordinates import getElementCoordinates, getElementDisplacements
from elements import Element
from B_mat import  get_B_plane_stress
import numpy as np

def calcStressStrainElemental(d,A,NodalCoord,Connectivity):
    #calculate strains in center of element!
    E = Element("Q4")
    stress = np.zeros((3,len(Connectivity)))
    strain = np.zeros((3,len(Connectivity)))
    for e in range(0,len(Connectivity)):
        Coord_mat_el = getElementCoordinates(e,NodalCoord,Connectivity)
        Elemental_disp = getElementDisplacements(e,d,A)
        gradN,detJ = E.G_map(0, 0,Coord_mat_el)
        B = get_B_plane_stress(gradN)
        strain[:,e] = np.dot(B,Elemental_disp)[:,0]
        stress[:,e] = strain[:,e]
    return strain,stress




def getStrain(NodalCoord,Connectivity,total_disp,D,pt,A):

    K = np.zeros((3,len(Connectivity)*4))  # K matrix will be num_dof x num_dof
    S = np.zeros((3,len(Connectivity)*4))
#we have to get all displacements back now

    for e in range(0,len(Connectivity)):
        Coord_mat_el = getElementCoordinates(e,NodalCoord,Connectivity)
        d_ele = getElementDisplacements(e,total_disp,A) #elemental disp
        #Loop over each element
        K[:,4*e:4*e+4] = getStrainElement(Coord_mat_el,d_ele)
        S[:,4*e:4*e+4] = getStressElement(K[:,4*e:4*e+4],D,pt)
    #end elemental loop
    return K,S

def getStrainElement(Coord_mat_el,d):
    E = Element("Q4")
    xIP=[-np.sqrt(3)/3,np.sqrt(3)/3]
    epsilon = np.zeros((3,4))
    count = 0
    for i in range(len(xIP)):
        for j in range(len(xIP)):
            gradN,detJ = E.G_map(xIP[i],xIP[j],Coord_mat_el)
            B = get_B_plane_stress(gradN)
            val = np.dot(B,d)
            for k in range(3):
                epsilon[k,count] = val[k]

            count += 1

    return epsilon

def getStressElement(Epsilon,D,pt):
    sigma = np.zeros((3,4))
    for i in range(4):
        sigma[:,i] = np.dot(D(Epsilon[:,i],pt),Epsilon[:,i])
        #print(sigma)
        #print( " ")
    return sigma

