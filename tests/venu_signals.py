# Kenneth Meyer
# 4/16/23
# script to run example DMD on synthetic data

import numpy as np

from sys import path
dmd_functions = "/home/kenneth/school/CSEM_class/machineLearning_datascience/CSE-382M-Project/src/"
path.append(dmd_functions)

# import functions from dmd_functions path
from DMD_methods import SpatioTemporalExample

if __name__ == "__main__":
    # functions from the matlab code (example)
    #  Define time and space discretizations
    # xi = linspace(-10,10,400);
    # t = linspace(0,4*pi,200); 
    # dt = t(2) - t(1);
    # [Xgrid,T] = meshgrid(xi,t);

    # Create two spatio-temporal patterns
    # f1 = sech (Xgrid+3) .* (1*exp(1j*2.3*T));
    # f2 = (sech (Xgrid) .*tanh(Xgrid)) .*(2*exp(1j*2.8*T));

    xi = np.linspace(-10,10,400)
    t = np.linspace(0,4*np.pi,200)
    #[Xgrid,T] = np.meshgrid(xi,t)

    # spatial-temporal functions to test!!
    def f1(Xi,T): 
        # component-wise multiplication!!
        #return np.multiply(1/np.cosh(Xi+3),(1*np.exp(1j*2.3*T)))
        return np.multiply(Xi**2,1*np.exp(1j*2.3*T))
    
    def f2(Xi,T):
        #return np.multiply((np.multiply((1/np.cosh(Xi)),(np.arctan(Xi)))), (2*np.exp(1j*2.8*T)))
        return np.multiply(np.tan(Xi),2*np.exp(1j*2.8*T))
    
    dmd_ex = SpatioTemporalExample(xi,t,f1,f2) # define object
    dmd_ex.plot_patterns(type='3d') # plots frequencies ( should show...)

    # also need to run DMD on this; run DMD on the 2D example to check that the method is working as intended!!!