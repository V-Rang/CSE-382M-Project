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

# # Visualize f1, f2, and f
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(Xgrid, T, np.real(f1), cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_title(r'$f_1 = {x^2}e^{j2.3t}$',fontsize=20)

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot_surface(Xgrid, T, np.real(f2), cmap='viridis')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_title(r'$f_2 = \tan(x)2e^{j2.8t}$',fontsize=20)

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot_surface(Xgrid, T, np.real(f), cmap='viridis')
ax3.set_xlabel('x')
ax3.set_ylabel('t')
ax3.set_title(r'$f = f_1 + f_2$',fontsize=20)


# plt.show()
def rSVD(X,r,q,p):
    #p = 0
    #q = 1
    ny = X.shape[1]
    P = np.random.randn(ny,r+p)
    Z = X @ P
    for k in range(q):
        Z = X @(X.T @ Z)
    Q,R = np.linalg.qr(Z,mode='reduced')

    Y = Q.T@X
    UY,S,VT = np.linalg.svd(Y,full_matrices=False)
    U = Q @ UY

    return U,S,VT

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
    


def DMD(X,num_modes,method=None,q=0,p=0):
    dt = 0.06314759102693052
    X1 = X[:,:-1]
    X2 = X[:,1:]
    if method == "rSVD":
        U,S,VT = rSVD(X1,num_modes,q,p)
    else:
        U,S,VT = np.linalg.svd(X1)
    #print(VT[0][0])
    #print(S[0])
    singular_vals = S
    V = VT.T
    S = np.diag(S)
    #print(U.shape[1])
    Ur = U[:,:num_modes]
    Sr = S[:num_modes,:num_modes]
    Vr = np.conj(V[:,:num_modes])
    #Vr = V[:,:num_modes]
    # ^ not sure what's going on here

    Atilde = Ur.T@X2@Vr@np.linalg.inv(Sr)

    # I think something is going on here...
    [eigvals,eigvecs] = np.linalg.eig(Atilde)
    Phi = X2@Vr@np.linalg.inv(Sr)@eigvecs
    omega = np.zeros(len(eigvals),dtype=complex)
    for ind in range(len(eigvals)):
        omega[ind] = cmath.log(eigvals[ind],math.e)/dt
    return eigvals,Phi,omega,singular_vals

num_modes=2

# X1 = X[:,:-1]
# X2 = X[:,1:]
# U,S,VT = scipy.linalg.svd(X1)
# num_modes = 2
# V = VT.T
# S = np.diag(S)
# Ur = U[:,:num_modes]
# Sr = S[:num_modes,:num_modes]
# Vr = np.conj(V[:,:num_modes])

# Atilde = Ur.T@X2@Vr@np.linalg.inv(Sr)
# [eigvals,eigvecs] = np.linalg.eig(Atilde)
# Phi = X2@Vr@np.linalg.inv(Sr)@eigvecs
# omega = np.zeros(len(eigvals),dtype=complex)
# for ind in range(len(eigvals)):
#         omega[ind] = cmath.log(eigvals[ind],math.e)/dt

# print(omega)
# print(Vr[:4,:],'\n')
# print(Vr2[:4,:])

eigvals,Phi,omega,aa = DMD(X,2)

# print(eigvals)
# print(omegavals)

#Predictive reconstruction
b = np.linalg.pinv(Phi)@X[:,0]

time_dynamics = np.zeros( (num_modes,len(t)),dtype=complex )
for iter  in range(len(t)):
    time_dynamics[:,iter] =np.multiply(b,np.exp(omega*t[iter ]))

X_dmd = Phi@time_dynamics

#Attempt to extract 2 modes separately
#Mode1
# td_mode1 = np.zeros( (1,len(t)),dtype=complex )

# for iter in range(len(t)):
#      td_mode1[:,iter] = np.multiply(b[0],np.exp(omega[0]*t[iter ]))
    
# X_dmd1 = Phi[:,0].reshape(len(Phi[:,0]),1)@td_mode1
# print(Phi[:,0].shape)
# print(td_mode1.shape)

ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot_surface(Xgrid, T, np.real(X_dmd.T), cmap='viridis')
ax4.set_xlabel('x')
ax4.set_ylabel('t')
ax4.set_title('Reconstructed using 2 modes',fontsize=20)
plt.show()


td_mode1 = np.zeros( (1,len(t)),dtype=complex )

for iter in range(len(t)):
     td_mode1[:,iter] = np.multiply(b[0],np.exp(omega[0]*t[iter ]))
    
X_dmd1 = Phi[:,0].reshape(len(Phi[:,0]),1)@td_mode1



td_mode2 = np.zeros( (1,len(t)),dtype=complex )

for iter in range(len(t)):
     td_mode2[:,iter] = np.multiply(b[1],np.exp(omega[1]*t[iter]))
    
X_dmd2 = Phi[:,1].reshape(len(Phi[:,1]),1)@td_mode1


### examination of the eigenvalues computed using randomSVD.


# fig = plt.figure(figsize=(10, 10))

# ax5 = fig.add_subplot(2, 2, 1, projection='3d')
# ax5.plot_surface(Xgrid, T, np.real(X_dmd1.T), cmap='viridis')
# ax5.set_xlabel('x')
# ax5.set_ylabel('t')
# ax5.set_title('Mode 1',fontsize=20)
# # plt.show()

# ax6 = fig.add_subplot(2, 2, 2, projection='3d')
# ax6.plot_surface(Xgrid, T, np.real(X_dmd2.T), cmap='viridis')
# ax6.set_xlabel('x')
# ax6.set_ylabel('t')
# ax6.set_title('Mode 2',fontsize=20)
# # plt.show()

# ax1 = fig.add_subplot(2, 2, 3, projection='3d')
# ax1.plot_surface(Xgrid, T, np.real(f1), cmap='viridis')
# ax1.set_xlabel('x')
# ax1.set_ylabel('t')
# ax1.set_title(r'$f_1 = {x^2}e^{j2.3t}$',fontsize=20)

# ax2 = fig.add_subplot(2, 2, 4, projection='3d')
# ax2.plot_surface(Xgrid, T, np.real(f2), cmap='viridis')
# ax2.set_xlabel('x')
# ax2.set_ylabel('t')
# ax2.set_title(r'$f_2 = \tan(x)2e^{j2.8t}$',fontsize=20)
# plt.savefig('singular_val_figs/testing.png')
# plt.show()


## extract more information regarding the singular values of the plots
SVD_singular_vals = []
dmd_modes = []
dmd_frequencies = []
eigenvalues = []
for i in range(0,10):
    eigvals,Phi,omega,singular_vals = DMD(X,num_modes=2,method="rSVd",q=0,p=3)
    SVD_singular_vals.append(singular_vals)
    dmd_modes.append(Phi)
    dmd_frequencies.append(omega)
    eigenvalues.append(eigvals)

### now, run some plots and statistics on these bad boys.
mode_means = np.mean(dmd_modes,axis=0)
mode_std = np.std(dmd_modes,axis=0)
SVD_singular_vals_means = np.mean(SVD_singular_vals,axis=0)
SVD_singular_vals_std = np.std(SVD_singular_vals,axis=0)
dmd_frequency_means = np.mean(np.imag(dmd_frequencies),axis=0)
dmd_frequency_std = np.std(np.imag(dmd_frequencies),axis=0)

eigenvalue_means = np.mean(eigenvalues,axis=0)
eigenvalue_std = np.std(eigenvalues,axis=0)

print("printing some means and standard deviations of the data")
print(f"Mean of singular values is: {SVD_singular_vals_means}")
print(f"Standard Deviation of singular values is: {SVD_singular_vals_std}")

print(f"Mean of Extracted Frequencies is: {dmd_frequency_means}")
print(f"Standard Deviation of Extracted Frequencies is: {dmd_frequency_std}")

dmd_frequencies = np.array(np.imag(dmd_frequencies))
second_freqs = dmd_frequencies[:,1]
print(f"Frequencies of second mode:{second_freqs}")
#print(f"Comparison of singular values to Actual singular values:{}")

# A = np.random.normal(size=(10,10))
# for k in range(10):
#     U,S,VT = np.linalg.svd(A)
#     #print(VT[0])


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