# Venu's code to make quiver and contour plots
# 4/10/23
# CSE 382M final project


#time step = 0.01

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
#fn = '/workspace/MLDS_Project/cylinder2d_nc/cylinder2d.nc'
fn = 'data/cylinder2d.nc'
ds = nc.Dataset(fn)
# print(ds)
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
# cp = ax.contourf(X, Y, velocity) # contour plot with lines
# #cp = ax.contour(X, Y, velocity) # filled contour plot
# cbar = plt.colorbar(cp)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Velocity Magnitude')
# plt.show()
#############################################
#Attempt to make movie
# from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from matplotlib import animation
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
fig,ax = plt.subplots()
x = np.linspace(-0.5,7.5, 640)
y = np.linspace(-0.5,0.5, 80)
X, Y = np.meshgrid(x, y)

def update(i):
    velocity = np.sqrt(uvalues[i]**2 + vvalues[i]**2)
    cp = ax.contourf(X, Y, velocity)
    #cbar = plt.colorbar(cp)
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_title('Velocity Magnitude')

anim = animation.FuncAnimation(fig,update,frames=1501,interval=2,blit=False)
plt.show()




#######################Test Animation###########################
# from matplotlib import pyplot as plt
# from matplotlib import animation
# fig = plt.figure()
# ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
# line, = ax.plot([], [], lw=2)

# # initialization function: plot the background of each frame
# def init():
#     line.set_data([], [])
#     return line,

# # animation function.  This is called sequentially
# def animate(i):
#     x = np.linspace(0, 2, 1000)
#     y = np.sin(2 * np.pi * (x - 0.01 * i))
#     line.set_data(x, y)
#     return line,

# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=200, interval=20, blit=True)

# # save the animation as an mp4.  This requires ffmpeg or mencoder to be
# # installed.  The extra_args ensure that the x264 codec is used, so that
# # the video can be embedded in html5.  You may need to adjust this for
# # your system: for more information, see
# # http://matplotlib.sourceforge.net/api/animation_api.html
# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# plt.show()







##############################################################








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

