# Kenneth Meyer
# 4/16/23
# script to run DMD on example image (2D) data!!!

import numpy as np

from sys import path
dmd_functions = "/home/kenneth/school/CSEM_class/machineLearning_datascience/CSE-382M-Project/src/"
path.append(dmd_functions)

# import functions from dmd_functions path
from DMD_methods import SpatioTemporalExample,ImageExample

# definition of functions that generate spatial-temporal patterns

def f_1(X,Y):
    """
        spatial-temporal function to compute DMD on.

        Inputs:
        X : spatial data from np.meshgrid()
        Y : ""
    """
    return np.exp(-1*((X - 40)**2)/250 + (Y-40)**2/250)

def f_2(X,Y):
    """
        Another spatial-temporal function to compute DMD on
    
    """
    # makes use of pixel structure to generate data
    nx,ny = X.shape
    mode = np.zeros((nx,ny))
    mode[nx-40:nx-10,ny-40,ny-10] = 1
    return mode

def lmbda(f,dt):
    return np.exp(1j*f*2*np.pi*dt)
    
## NOTE: this might be for multiresolution DMD; might need to alter code a bit???
    
if __name__ == "__main__":
    # eventually turn this into a class/file we can use with yaml if desired
    nx = 80
    ny = 80
    T = 10 # seconds
    dt = 0.01
    noise = 0.01
    # fequency, amplitude, and "range" of each function
    params = {"f1":[5.55,1,[0,5],lmbda],"f2":[0.9,1,[3,7],lmbda],"f3":[0.15,0.5,[0,T],lmbda]} 

    im_test_dmd  = ImageExample(nx,ny,T,dt,**params)
    im_test_dmd.generate_movie()

    # ^ need to use another class.

    


