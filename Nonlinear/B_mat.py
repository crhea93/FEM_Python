#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 19:50:20 2017

@author: crhea
"""
import numpy as np
def get_B_plane_stress(gradN):
    (n,m) = gradN.shape
    B = np.zeros((3,8))
    for i in range(0,4):
        B[0, 2 * (i)] = gradN[0, i]
        B[1, 2 * (i)+1] = gradN[1, i]
        B[2, 2 * (i)] = gradN[1, i]
        B[2, 2 * (i)+1] = gradN[0, i]
 
    return B
    
