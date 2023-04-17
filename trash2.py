import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import cmath

def rSVD(X,r,q,p):
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

def DMD(X,num_modes):
    dt = 1/100
    X1 = X[:,:-1]
    X2 = X[:,1:]
    # U,S,VT = np.linalg.svd(X1)
    # U,S,VT = rSVD(X1,num_modes,5,0)
    U,S,VT = np.linalg.svd(X1)
    # plt.scatter(np.arange(len(S)),S)
    # plt.show()
    V = VT.T
    S = np.diag(S)
    Ur = U[:,:num_modes]
    Sr = S[:num_modes,:num_modes]
    Vr = V[:,:num_modes]
    Atilde = Ur.T@X2@Vr@np.linalg.inv(Sr)
    [eigvals,eigvecs] = np.linalg.eig(Atilde)
    # return eigvals
    # idx = eigvals.argsort()[::-1]
    # eigvals = eigvals[idx]
    # eigvecs = eigvecs[idx]
    Phi = X2@Vr@np.linalg.inv(Sr)@eigvecs
    omega = []
    for value in eigvals:
        omega.append(cmath.log(value,math.e)/dt)
    return eigvals,Phi,np.array(omega)


nx = 100
ny = 100

Tlim = 10 # seconds
dt = 0.01
t = np.arange(0, Tlim, dt)

T = np.arange(0,Tlim,dt)


# Define the 3D matrix Xclean
Xclean = np.ones((len(t), ny,nx))
    
x, y = np.meshgrid(np.arange(nx), np.arange(ny))
for i in range(len(t)):
    Xclean[i, :, :] -= 0.5 + 0.5 * np.cos(2 * np.pi * t[i] * 10)*(np.sqrt((x-nx/2)**2 + (y-ny/2)**2) < 25)  #10Hz

    # Xclean[i, :, :] -= 0.5 + 0.5 * np.cos(t[i] * 10) * (np.sqrt((x-nx/2)**2 + (y-ny/2)**2) < 25) #5Hz


# Add the square blinking pattern
for i in range(len(t)):
    Xclean[i, 20:40, 20:40] -= 0.5 + 0.5 * np.cos(2 * np.pi * t[i] * 10) #10Hz
    # Xclean[i, 20:40, 20:40] -= 0.5 + 0.5 * np.cos(t[i] * 100) #10Hz


#Works
fig, ax = plt.subplots()
# fig.set_figheight(10000)
# fig.set_figwidth(25)
# fig = plt.figure()
# ax = plt.axes(xlim=(0,7.5),ylim=(-0.5,0.5))

im = ax.imshow(Xclean[0].real,aspect='equal',cmap='viridis')
fig.colorbar(im)

ax.set_xticks( np.linspace(0,Xclean[0].shape[1],6,dtype=int) ,np.linspace(0,100,6) )
ax.set_yticks( np.linspace(Xclean[0].shape[0],0,6,dtype=int) ,np.linspace(0,100,6) )

# ax.set_title('von Karman Vortex Street')

# # Define the update function for the animation
def update(frame):
    im.set_data(Xclean[frame].real)
    return [im]


# # Create the animation
ani = animation.FuncAnimation(fig, update, frames=Xclean.shape[0], interval=100, blit=True)
fig.patch.set_facecolor('white')

# # Display the animation
plt.show()

xdim = len(np.arange(nx))
ydim = len(np.arange(ny))
tdim = len((t))

X = np.zeros((xdim*ydim,tdim))  #snapshots matrix
for t in range(tdim):
    X[:,t] = Xclean[t].reshape(xdim*ydim)

# X1 = X[:,:-1]
# X2 = X[:,1:]

eigvals,Phi,omegavals = DMD(X,3)  #real part of each omegavals is the growth rate of the mode and imaginary part of each omegaval is the frequency
intial_amps = np.linalg.pinv(Phi)@X[:,0]


# print(eigvals)
print(omegavals/(2*np.pi))
# print(X.shape)



#Some other mode
index = 2
mode_eigval = eigvals[index]
mode_init = Phi[:,index]
mode_init_org_shape = mode_init.reshape((ydim,xdim))
modes_through_time = np.zeros((len(T),ydim,xdim),dtype=complex)
# modes_through_time[0] = mode_init_org_shape


for t in range(1,len(T)):
    modes_through_time[t] = mode_init_org_shape**intial_amps[index]*cmath.exp(omegavals[index]*T[t])


# print(uvalues.shape,bg_modes_through_time.shape)

#fig, ax = plt.subplots()
#im = ax.imshow(modes_through_time[0].real,aspect='equal',cmap='viridis')
#fig.colorbar(im)

# ax.set_xticks( np.linspace(0,Xclean[0].shape[1],6,dtype=int) ,np.linspace(0,100,6) )
# ax.set_yticks( np.linspace(Xclean[0].shape[0],0,6,dtype=int) ,np.linspace(0,100,6) )

# ax.set_title(f'DMD Mode {index}, frequency = {"%.2f"%(omegavals[index].imag Hz, growth rate = {"%.2f"%(omegavals[index].real)/(2*np.pi)}Hz')

#def update(frame):
#    im.set_data(modes_through_time[frame].real)
#    return [im]

#ani = animation.FuncAnimation(fig, update, frames=modes_through_time.shape[0], interval=2, blit=True)
#fig.patch.set_facecolor('white')

#writervideo = animation.PillowWriter(fps=60)

#ani.save('testing.gif', writer=writervideo)

plt.show()
