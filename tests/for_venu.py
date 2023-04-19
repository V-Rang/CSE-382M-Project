# Kenneth Meyer
# 4/18/23
# plotting singular values of the systems I am running dmd on....

# below is code from VENU's test_calc.py example!
import numpy as np
import matplotlib.pyplot as plt
import cmath
import math
import scipy.linalg

from matplotlib import rc
rc('text',usetex=True)
# plt.rcParams(usetex = True)
xi = np.linspace(-10, 10, 400)
t = np.linspace(0, 4*np.pi, 200)
dt = t[1] - t[0]
print(dt)
Xgrid, T = np.meshgrid(xi, t)


# b = np.multiply(complex(0,2.3),T)
b = np.exp(1j*2.3*T)
a = Xgrid*Xgrid

f1 = np.multiply(a,b)

f2 = np.multiply(  np.tan(Xgrid), 2*np.exp(1j*2.8*T)   )


# f1 = np.multiply(1/np.cosh(Xgrid+3),1*np.exp(1j*2.3*T))
# f2 = np.multiply(1/np.cosh(Xgrid)*np.tanh(Xgrid), 2*np.exp(1j*2.8*T))

# # Create two spatiotemporal patterns
# # f1 = np.multiply(Xgrid.T,Xgrid.T)  , 1*cmath.exp(1j*2.3*T)   )
# f2 = np.tan(Xgrid) * (2*np.exp(1j*2.8*T))
f = f1 + f2

X = f.T  # Data Matrix


def rQB(X,r,p,q):
    """
    
        Computes random QB decomposition of a matrix X
    
    """

    # this is copy-pasted from the rSVD code above!
    ny = X.shape[1]
    P = np.random.randn(ny,r+p)
    Z = X @ P
    for k in range(q):
        Z = X @(np.conj(X.T) @ Z)
    #Q,R = np.linalg.qr(Z,mode='reduced')
    Q,R = np.linalg.qr(Z,mode='reduced')
    #print(Q[0,0])

    Y = np.conj(Q.T)@X

    return Q,Y

# make p at least 2 based on something I saw in the arxiv paper
def rDMD(X,num_modes,q=0,p=2):
    """
    
        Randomized DMD. Calls rSVD_QB, which computes the QB decomposition of the state X.

    """
    Q,B = rQB(X,num_modes,p,q)

    #print(Q.shape)
    #print(B.shape)

    # separate data into snapshots!
    BL = B[:,:-1]
    BR = B[:,1:]

    #print(BL.shape)
    #print(BR.shape)

    #U_tilde,S,VT = np.linalg.svd(BL,k) (truncated SVD) # <- psuedocode
    U_tilde,S,VT = np.linalg.svd(BL) # "truncated" SVD in the algo;

    # rank "k" trunctation of the SVD 
    U_tilde = U_tilde[:,:num_modes]
    S = S[:num_modes]
    S = np.diag(S)
    V = VT.T
    V = np.conj(V) # need hermitian because complex? check this
    V = V[:,:num_modes]

    # least squares fit - not sure if U should be hermitian or not
    #A_B = np.linalg.hermitian(U_tilde) @ BR @ V @ np.linalg.inv(S)
    A_B = np.conj(U_tilde.T) @ BR @ V @ np.linalg.inv(S)
    
    # eigenvalue decomposition
    [lmbda,W_B] = np.linalg.eig(A_B)
    # recovered high-dimensional modes
    W = Q @ BR @ V @ np.linalg.inv(S) @ W_B

    return W,lmbda

print(f"----------------------------------------------")
num_modes=2
print(X.shape)
dt = 0.06314759102693052
for i in range(1):
    # W : modes
    # lmbda : eigenvalues
    W,lmbda = rDMD(X,num_modes,q=0,p=4)
    #print(W.shape)# not sure what this is rn
    #print(np.log(np.max(np.real(lmbda))))
    print(len(lmbda))
    x = 0
    y = 100000
    for l in lmbda:
        temp = np.imag(cmath.log(l,math.e)/dt)
        print(temp)
    #print(np.max(np.real(lmbda)))