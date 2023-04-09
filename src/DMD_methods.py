# Kenneth Meyer and Venu R.
# 4/8/23
# CSE 382M final project

## functions that perform different types of DMD!

## consider turning this into a class so functions for data analysis are easily callable?..

import numpy as np
import math

def DMD(X,num_modes,dt):
    """
        Computes modes and associated eigenvalues and eigenvectors of spatial-temporal data using
        dynamic mode decomposition (DMD)

        Algorithm 1.1 from (DMD) textbook.

        PARAMETERS
        ----------
        X : data
        num_modes : r in algo 1.1, desired rank of the system (number of modes)
        dt : time step
        ^ might be able to vary this?

        OUTPUTS
        -------
    
    """

    X1 = X[:,:-1]
    X2 = X[:,1:]
    U,S,VT = np.linalg.svd(X1)
    V = VT.T
    S = np.diag(S)
    Ur = U[:,:num_modes]
    Sr = S[:num_modes,:num_modes]
    Vr = V[:,:num_modes]


    Atilde = Ur.T@X2@Vr@np.linalg.inv(Sr)
    [eigvals,eigvecs] = np.linalg.eig(Atilde)
    eigvals = np.diag(eigvals)
    Phi = X2@Vr@np.linalg.inv(Sr)@eigvecs
    lam = np.diag(eigvals)
    omega = []
    for value in lam:
        omega.append(math.log(value,math.e)/dt)

    #Computing DMD solution
    x1 = X[:,1]
    b = np.linalg.pinv(Phi)@x1
    mm1 = X1.shape[1]

    time_dynamics = np.zeros( (num_modes,mm1) )
    t = np.arange(0,mm1)*dt

    # print(time_dynamics,'\n')

    for iter in range(0,mm1):
        time_dynamics[:,iter] = np.multiply(omega,t[iter])

    Xdmd = Phi@time_dynamics

    # returning 4 variables makes me think a class could be beneficial
    data = {}
    data["Phi"] = Phi
    data["omega"] = omega
    data["lam"] = lam
    data["Xdmd"] = Xdmd
    
    return data