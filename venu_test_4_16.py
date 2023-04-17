#time step = 0.01

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cmath
import matplotlib.animation as animation

fn = '/workspace/MLDS_Project/cylinder2d_nc/cylinder2d.nc'
ds = nc.Dataset(fn)
# print(ds.variables['tdim'])
uvel = ds.variables['u']
vvel = ds.variables['v']
uvalues = uvel[:]
vvalues = vvel[:]
# print(uvalues.shape)
# print(vvalues.shape)

#attempt to plot for single timestep - quiver plot and contour plot
x = np.linspace(-0.5,7.5, 640)
y = np.linspace(-0.5,0.5, 80)
X, Y = np.meshgrid(x, y)
umat1 = uvalues[0]
vmat1 = vvalues[0]
#quiver plot
# fig, ax = plt.subplots()
# ax.quiver(X, Y, umat1,vmat1)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Velocity Field')
# plt.show()
########################################################
#contour plot
# velocity = np.sqrt(umat1**2 + vmat1**2)
# fig, ax = plt.subplots()
# cp = ax.contourf(X, Y, velocity)
# cbar = plt.colorbar(cp)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Velocity Magnitude')
# plt.show()
#############################################
#Attempt to make movie
# from matplotlib.animation import FuncAnimation
# from matplotlib import pyplot as plt
# from matplotlib import animation
timevalues = ds.variables['tdim']
T = timevalues[:]


#Attempt1
# fig = plt.figure()
# ax = plt.axes(xlim=(0,7.5),ylim=(-0.5,0.5))

# line, = ax.plot([],lw=2)

# def init():
#     line.set_data([])
#     return line,

# def animate(i):
#     umat = uvalues[i]
#     vmat = vvalues[i]
#     line.set_data(np.sqrt(umat**2 + vmat**2))

#     return line,

# T = timevalues[:]
# anim = animation.FuncAnimation(fig,animate,init_func=init,frames=1501,interval=20,blit=True)
# plt.show()

#Attempt2

# fig,ax = plt.subplots()
# x = np.linspace(-0.5,7.5, 640)
# y = np.linspace(-0.5,0.5, 80)
# X, Y = np.meshgrid(x, y)

# def update(i):
#     velocity = np.sqrt(uvalues[i]**2 + vvalues[i]**2)
#     cp = ax.contourf(X, Y, velocity)
#     cbar = plt.colorbar(cp)
#     # ax.set_xlabel('X')
#     # ax.set_ylabel('Y')
#     # ax.set_title('Velocity Magnitude')

# anim = animation.FuncAnimation(fig,update,frames=1501,interval=20,blit=False)
# plt.show()

#Attempt3
#working
#only x component of velocity
########################################################################
# import matplotlib.pyplot as plt

# print(uvalues[0].shape)
# fig, ax = plt.subplots()
# # fig.set_figheight(10000)
# # fig.set_figwidth(25)
# # fig = plt.figure()
# # ax = plt.axes(xlim=(0,7.5),ylim=(-0.5,0.5))

# im = ax.imshow(uvalues[0],aspect='auto',cmap='viridis')
# fig.colorbar(im)

# ax.set_xticks( np.linspace(0,uvalues[0].shape[1],9,dtype=int) ,np.linspace(-0.5,7.5,9) )
# ax.set_yticks( np.linspace(uvalues[0].shape[0],0,3,dtype=int) ,np.linspace(0.5,-0.5,3) )


# ax.set_title('von Karman Vortex Street')

# # Define the update function for the animation
# def update(frame):
#     im.set_data(uvalues[frame])
#     return [im]

# # Create the animation
# ani = animation.FuncAnimation(fig, update, frames=uvalues.shape[0], interval=5, blit=True)

# # Display the animation
# plt.show()
##############################################################

# #Attempt4
#total velocity u**2 + v**2
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# print(uvalues[0].shape)
# fig, ax = plt.subplots()

# im = ax.imshow(uvalues[0]**2 + vvalues[0]**2,aspect='equal',cmap='viridis')
# fig.colorbar(im)

# ax.set_xticks( np.linspace(0,uvalues[0].shape[1],9,dtype=int) ,np.linspace(-0.5,7.5,9) )
# ax.set_yticks( np.linspace(uvalues[0].shape[0],0,3,dtype=int) ,np.linspace(0.5,-0.5,3) )


# ax.set_title('von Karman Vortex Street')

# # Define the update function for the animation
# def update(frame):
#     im.set_data(uvalues[frame])
#     return [im]

# # Create the animation
# ani = animation.FuncAnimation(fig, update, frames=uvalues.shape[0], interval=5, blit=True)

# Display the animation
# plt.show()
##################################################################
#Attempt5
# only y comp of velocity v
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# print(uvalues[0].shape)
# fig, ax = plt.subplots()

# im = ax.imshow(vvalues[0],aspect='equal',cmap='viridis')
# fig.colorbar(im)

# ax.set_xticks( np.linspace(0,uvalues[0].shape[1],9,dtype=int) ,np.linspace(-0.5,7.5,9) )
# ax.set_yticks( np.linspace(uvalues[0].shape[0],0,3,dtype=int) ,np.linspace(0.5,-0.5,3) )


# ax.set_title('von Karman Vortex Street')

# # Define the update function for the animation
# def update(frame):
#     im.set_data(uvalues[frame])
#     return [im]

# # Create the animation
# ani = animation.FuncAnimation(fig, update, frames=uvalues.shape[0], interval=5, blit=True)

# # Display the animation
# plt.show()

###########################################################################

# print(ds.variables)
# for i in ds.variables.keys():
    # print(i)

# ans = ds.variables['tdim']
# xvals = ans[:]
# print(xvals)

# visc = ds.variables['nu']
# viscval = visc[:]
# print(viscval)
# print(uvalues)
# print(type(uvalues))

# print(uvalues.shape)

# u_mat = []

# sample_uvals = uvalues[1500]
# x = np.linspace(-0.5,7.5, 640)
# y = np.linspace(-0.5,0.5, 80)
# X, Y = np.meshgrid(x, y)
# fig, ax = plt.subplots()
# cp = ax.contourf(X, Y, sample_uvals)

# # Add a colorbar
# cbar = plt.colorbar(cp)

# # Customize the plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Velocity Magnitude')

# # Show the plot
# plt.show()


# print(sample_uvals)

# import numpy as np
# import matplotlib.pyplot as plt

# # Generate some example data
# x = np.linspace(-1, 1, 100)
# y = np.linspace(-1, 1, 100)
# X, Y = np.meshgrid(x, y)
# # U = np.sin(np.pi*X)*np.cos(np.pi*Y)
# # V = -np.cos(np.pi*X)*np.sin(np.pi*Y)

# velocity = np.sqrt(U**2 + V**2)

# # Create a contour plot
# fig, ax = plt.subplots()
# cp = ax.contourf(X, Y, velocity)

# # Add a colorbar
# cbar = plt.colorbar(cp)

# # Customize the plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Velocity Magnitude')

# # Show the plot
# plt.show()


########DMD of flow data
xdim = len(ds.variables['xdim'][:])
ydim = len(ds.variables['ydim'][:])
tdim = len(ds.variables['tdim'][:])

X = np.zeros((xdim*ydim,tdim))  #snapshots matrix

for t in range(tdim):
    X[:,t] = uvalues[t].reshape(xdim*ydim)


#determining how many modes to extract by plotting singular values
X1 = X[:,:-1]
X2 = X[:,1:]

#randomized SVD
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

# U,S,VT = rSVD(X1,1000,4,5)
# fig,ax = plt.subplots()
# plt.scatter(np.arange(len(S)),S)
# plt.show() #=> only 1 mode needed

# print(X1.shape)
import dask.array as da
import math
# import scipy
# from scipy.linalg import svd

# k = 100
# print(s)
# U,S,VT = da.linalg.svd(da(X1)) #crashes svd of 51200 X 1501 matrix
# plt.plot(S)
###################
#function
def DMD(X,num_modes):
    dt = 1/100
    X1 = X[:,:-1]
    X2 = X[:,1:]
    # U,S,VT = np.linalg.svd(X1)
    U,S,VT = rSVD(X1,num_modes,3,0)
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

# print(eigvals)
eigvals,Phi,omegavals = DMD(X,25)  #real part of each omegavals is the growth rate of the mode and imaginary part of each omegaval is the frequency
intial_amps = np.linalg.pinv(Phi)@X[:,0]

#this plot works
########################################################
# print(eigvals)
# theta = np.linspace(0,2*np.pi,1000)
# x = 1*np.cos(theta)
# y = 1*np.sin(theta)
# plt.plot(x,y)
# plt.title('Eigenvalue spectrum')
# lims = np.abs(eigvals) 
# plt.axis('equal')
# plt.scatter(eigvals[lims>1].real,eigvals[lims>1].imag,c='r',label='Outside')
# plt.scatter(eigvals[lims==1].real,eigvals[lims==1].imag,c='b',label='On')
# plt.scatter(eigvals[lims<1].real,eigvals[lims<1].imag,c='g',label='Inside')
# plt.xlabel('Real Component')
# plt.ylabel('Imaginary Component')
# plt.legend()
# plt.show()
################################################
#plotting DMD modes
#background mode
bg_index = [i for i in range(len(eigvals)) if (abs(eigvals[i].real -1)<1/10000 and eigvals[i].imag == 0)][0]
bg_eigval = eigvals[bg_index]

bg_mode_init = Phi[:,bg_index]
bg_mode_init_org_shape = bg_mode_init.reshape((ydim,xdim))
bg_modes_through_time = np.zeros((len(T),ydim,xdim))
bg_modes_through_time[0] = bg_mode_init_org_shape


for t in range(1,len(T)):
    bg_modes_through_time[t] = bg_mode_init_org_shape*cmath.exp(omegavals[bg_index]*T[t])*abs(intial_amps[bg_index])


# print(uvalues.shape,bg_modes_through_time.shape)


# fig, ax = plt.subplots()
# im = ax.imshow(bg_modes_through_time[0],aspect='equal',cmap='viridis')
# fig.colorbar(im)

# ax.set_xticks( np.linspace(0,bg_modes_through_time[0].shape[1],9,dtype=int) ,np.linspace(-0.5,7.5,9) )
# ax.set_yticks( np.linspace(bg_modes_through_time[0].shape[0],0,3,dtype=int) ,np.linspace(0.5,-0.5,3) )

# ax.set_title(f'DMD Mode {bg_index}, frequency = {"%.2f"%omegavals[bg_index].imag}Hz, growth rate = {"%.2f"%omegavals[bg_index].real}Hz')
# def update(frame):
#     im.set_data(bg_modes_through_time[frame])
#     return [im]

# ani = animation.FuncAnimation(fig, update, frames=bg_modes_through_time.shape[0], interval=5, blit=True)
# plt.show()

##########################################################################

#Some other mode

index = 15
mode_eigval = eigvals[index]

mode_init = Phi[:,index]
mode_init_org_shape = mode_init.reshape((ydim,xdim))
modes_through_time = np.zeros((len(T),ydim,xdim))
modes_through_time[0] = mode_init_org_shape


for t in range(1,len(T)):
    modes_through_time[t] = mode_init_org_shape*cmath.exp(omegavals[index]*T[t])*abs(intial_amps[index])


# print(uvalues.shape,bg_modes_through_time.shape)


fig, ax = plt.subplots()
im = ax.imshow(modes_through_time[0],aspect='equal',cmap='viridis')
fig.colorbar(im)

ax.set_xticks( np.linspace(0,modes_through_time[0].shape[1],9,dtype=int) ,np.linspace(-0.5,7.5,9) )
ax.set_yticks( np.linspace(modes_through_time[0].shape[0],0,3,dtype=int) ,np.linspace(0.5,-0.5,3) )


ax.set_title(f'DMD Mode {index}, frequency = {"%.2f"%omegavals[index].imag}Hz, growth rate = {"%.2f"%omegavals[index].real}Hz')

def update(frame):
    im.set_data(modes_through_time[frame])
    return [im]

ani = animation.FuncAnimation(fig, update, frames=modes_through_time.shape[0], interval=2, blit=True)
plt.show()


#Predictive Reconstruction using x number of modes
desired_num = 5 #can change
eigvals,Phi,omegavals = DMD(X,desired_num)  #real part of each omegavals is the growth rate of the mode and imaginary part of each omegaval is the frequency
intial_amps = np.linalg.pinv(Phi)@X[:,0]

# sample = np.zeros((ydim,xdim),dtype=complex)
# print(Phi[:,0].reshape((ydim,xdim)).shape)

# for mode_num in range(5):
#     sample += Phi[:,mode_num].reshape((ydim,xdim))*cmath.exp(omegavals[mode_num]*T[t])*abs(intial_amps[mode_num])

# sample = Phi[:,0].reshape((ydim,xdim))*cmath.exp(omegavals[0]*T[t])*abs(intial_amps[0])
# print(sample.shape)
########fix this:
# reconstr = np.zeros( (len(T),ydim,xdim),dtype=complex)
# for t in range(len(T)):
#     for mode_num in range(desired_num):
#         reconstr[t] += Phi[:,mode_num].reshape((ydim,xdim))*cmath.exp(omegavals[mode_num]*T[t])*abs(intial_amps[mode_num])


# fig, ax = plt.subplots()
# im = ax.imshow(reconstr[0],aspect='equal',cmap='viridis')
# fig.colorbar(im)

# ax.set_xticks( np.linspace(0,modes_through_time[0].shape[1],9,dtype=int) ,np.linspace(-0.5,7.5,9) )
# ax.set_yticks( np.linspace(modes_through_time[0].shape[0],0,3,dtype=int) ,np.linspace(0.5,-0.5,3) )


# ax.set_title('Reconstructed plot')

# def update(frame):
#     im.set_data(reconstr[frame])
#     return [im]

# ani = animation.FuncAnimation(fig, update, frames=reconstr.shape[0], interval=2, blit=True)
# plt.show()







###########################################################################

# print(intial_amps.shape)
# print(background_index)

# plt.xlim((0.8,1.2))
# print(eigvals[lims>=1],lims[lims>=1])


# U,S,VT = rSVD(X1,20,1,5)
# plt.scatter(np.arange(21),S[:21])
# plt.show()




#FFT

# v = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# # Perform FFT on the velocity data
# fft = np.fft.fft(v)

# # Calculate the frequency axis
# freq = np.fft.fftfreq(len(v), d=t[1]-t[0])

# # Plot the frequency spectrum
# plt.plot(freq, abs(fft))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.show()
