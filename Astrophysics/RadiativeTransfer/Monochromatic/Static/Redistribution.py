'''Redistribution Functions '''
import numpy as np
def RedistributionFunc(g,theta):
    val = (1-g**2)/(1+g**2-2*g*np.cos(theta[0]))**(3/2)
    return (1/(4*np.pi))*val
