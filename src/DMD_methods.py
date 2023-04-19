# Kenneth Meyer and Venu R.
# 4/8/23
# CSE 382M final project

## functions that perform different types of DMD!

## consider turning this into a class so functions for data analysis are easily callable?..

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.animation as animation
#from animation import FFMpegwriter

#animation.FFMpegWriter

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

### ^ this was written last week, there have been updates made to the DMD code, need to check with Venu
####    and make a new function if we want to use a function.

# THE TWO CLASSES BELOW SHOULD INHERIT A DMD CLASS

# making a new class for images, diverging too much from the first example.
class ImageExample:
    def __init__(self,nx,ny,T,dt,**kwargs):
        """
        
            Initialization of a DMD example for an image.

            kwargs contains all of the different modes; potentially in a class.
        
        """

        self.nx = nx # vertical pixels
        self.ny = ny # horiz pixels
        self.n = self.nx*self.ny
        self.T = T   # seconds
        self.dt = dt

        self.t = np.arange(dt,T,dt)

        # 2d example replicating the txtbook!
        self.X,self.Y = np.meshgrid(np.arange(1,self.nx+1),np.arange(1,self.ny+1))

        # save spatial-temporal patterns 
        self.patterns = [] # list of DATA
        self.funcs = []    # list of FUNCTIONS
        self.f = 0 # composite evaluation of functions

        # variables only used if there are kwargs, i.e. frequencies and amplitudes.
        self.freq = []
        self.amp = []
        self.range = []
        self.lmbda = []

        # horrible form but I'm going to hardcode these for now. can't think of another way.
        # functions defining each mode!
        self.f1 = np.exp(-1*((self.X - 40)**2)/250 + (self.Y-40)**2/250)
        self.f2 = np.zeros((self.nx,self.ny))
        self.f2[self.nx-40:self.nx-10,self.ny-40:self.ny-10] = 1
        self.f3 = np.zeros((self.nx,self.ny))
        self.f3[0:self.nx - 20,0:self.ny-20] = 1


        self.signals = [self.f1,self.f2,self.f3]

        #for f_i in args:
        #    # f(x_i,t) but from meshgrid
        #    f_i_eval = np.array(f_i(np.ravel(self.X),np.ravel(self.T)))
        #    f_i_eval = np.reshape(f_i_eval,self.X.shape)
        #    self.f += f_i_eval
        #    self.patterns.append(f_i_eval)
        #    self.funcs.append(f_i)

        # extract the frequency, amplitude, and range if given!
        for f_i_keys in kwargs:
            # format of data: 
            #   [0] -> frequency
            #   [1] -> amplitude
            #   [2] -> range
            freq_i = kwargs[f_i_keys][0]
            self.freq.append(freq_i)
            self.amp.append(kwargs[f_i_keys][1])
            self.range.append(kwargs[f_i_keys][2])
            self.lmbda.append(kwargs[f_i_keys][3](freq_i,self.dt)) # lmbda is a FUNCTION!!!

        print("frequencies:")
        print(self.freq)
    
        print("amplitudes:")
        print(self.amp)
        
        print("ranges:")
        print(self.range)
        
        print("lambda:")
        print(self.lmbda)


    def generate_movie(self):
        """

            Generates a movie for a 2D (image) example!
        
        """
        # might need to save some of these variables, looking through matlab code rn
        #Xclean = np.zeros((self.nx,self.ny,len(self.t)))

        # array for now
        Xclean = []

        # loop through each snapshot of time!
        # creating a mode class could have been a good idea...but this might work.
        for ti in range(0,len(self.t)):
            frame_data = np.zeros((self.nx,self.ny))
            # loop through each mode
            for jj in range(0,len(self.freq)):
                if ti > round(self.range[jj][0]/self.dt) and ti < round(self.range[jj][1]/self.dt):
                    frame_data += self.amp[jj] * self.signals[jj] * np.real(self.lmbda[jj]**(ti+1))

            #Xclean[:,:,ti] = frame
            Xclean.append(frame_data) # list of np arrays

        # add noise here, if desired.

        print(len(Xclean))
        
        # plot the image, standstill (use half for testing!)
        plt.imshow(Xclean[145])
        plt.savefig("figs/square_oval/static_square_oval.png")

        # plot the image
        #Works
        fig, ax = plt.subplots()
        # fig.set_figheight(10000)
        # fig.set_figwidth(25)
        # fig = plt.figure()
        # ax = plt.axes(xlim=(0,7.5),ylim=(-0.5,0.5))

        im = ax.imshow(np.real(Xclean[0]),aspect='equal',cmap='viridis')
        fig.colorbar(im)

        #ax.set_xticks( np.linspace(0,Xclean[0].shape[1],6,dtype=int) ,np.linspace(0,100,6) )
        #ax.set_yticks( np.linspace(Xclean[0].shape[0],0,6,dtype=int) ,np.linspace(0,100,6) )

        ax.set_title('von Karman Vortex Street')

        # # Define the update function for the animation
        def update(frame):
            im.set_data(np.real(Xclean[frame]))
            return [im]
        
        # # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(Xclean), interval=100, blit=True)
        #animation.F
        #fig.patch.set_facecolor('white')
        writervideo = animation.FFMpegWriter(fps=60)
        ani.save('increasingStraightLine.mp4', writer=writervideo)
        plt.close()

        # # Display the animation - only works in jupyter unfortunately
        plt.show()  

        ## save the animation

        


class SpatioTemporalExample:
    def __init__(self, xi, t: int, case="3d",dt=None, *args, **kwargs):
        """
        
            Initialization of a Spatial Temporal DMD example.

            Inputs:
            xi : space descritization (rectangular)
            t : time descritization (uniform)    
            *args : functions defining spatio-temporal patterns
    
        """

        self.xi = xi
        self.yi = 0
        # need nx and ny in image example...
        #self.nx = 0
        #self.ny = 0

        # check for the type of t (requires dt to be passed)
        #if type(t) == int:
        #    self.t = np.arange(dt,t,dt)
        #    self.dt = dt
        #else:
        #    # assumes t is an array
        
        self.t = t
        self.dt = t[1] - t[0]

        #self.nx = len(self.xi)
        #self.ny = len(self.yi)
        #self.n = self.nx * self.nt

        #if self.ny != 0:
        #    # 2d example replicating the txtbook!
        #    self.X,self.Y = np.meshgrid(self.xi,self.yi)
        #else:
        # 3d example from ch1 !!!
        self.X,self.T = np.meshgrid(self.xi,self.t) # is a tuple

        # save spatial-temporal patterns 
        self.patterns = [] # list of DATA
        self.funcs = []    # list of FUNCTIONS
        self.f = 0 # composite evaluation of functions

        # variables only used if there are kwargs, i.e. frequencies and amplitudes.
        self.freq = []
        self.amp = []
        self.range = []
        self.lmbda = []

        for f_i in args:
            # f(x_i,t) but from meshgrid
            f_i_eval = np.array(f_i(np.ravel(self.X),np.ravel(self.T)))
            f_i_eval = np.reshape(f_i_eval,self.X.shape)
            self.f += f_i_eval
            self.patterns.append(f_i_eval)
            self.funcs.append(f_i)

        # extract the frequency, amplitude, and range if given!
        #for f_i_keys in kwargs:
        #    # format of data: 
        #    #   [0] -> frequency
        #    #   [1] -> amplitude
        #    #   [2] -> range
        #    freq_i = kwargs[f_i_keys][0]
        #    self.freq.append(freq_i)
        #    self.amp.append(kwargs[f_i_keys][1])
        #    self.range.append(kwargs[f_i_keys][2])
        #    self.lmbda.append(kwargs[f_i_keys][3](freq_i,self.dt)) # lmbda is a FUNCTION!!!

        # compose all of the functions
        #self.f = 0
        # could use self.patterns, doesn't really matter.
        #for f_i in args:
        #    self.f += self.patterns[f_i]
        #     #self.f += f_i(self.xi,self.t)

        #self.f = self.f.T # transpose the signal (done in algo 1.2)

        
    def run_DMD(self):
        """
            Runs Dynamic Mode Decomposition on the example spatial-temporal data
        
        """


    def plot_patterns(self,type='3d'):
        """
        
            Plots individual signals (self.patterns) and combined signal (self.f)
        
        """

        if type=='3d':
            # case for surface data; sqaure-oval data might look different?
            i = 0
            print(len(self.patterns))
            for jj in range(0,len(self.patterns)):
                fig,ax = plt.subplots(subplot_kw={"projection":"3d"})
                #surf = ax.plot_surface(self.mesh[0],self.mesh[1],self.patterns[jj],cmap=cm.coolwarm,linewidth=0,antialiased=False)
                print(len(self.patterns[jj]))
                print(len(self.patterns[jj][0]))
                surf = ax.plot_surface(self.X,self.T,np.real(self.patterns[jj]),cmap=cm.coolwarm)
                fig.colorbar(surf,shrink=0.5,aspect=5)


                plt.show() # for use in jupyter notebook
                plt.savefig("figs/func_" + str(jj) + ".png") # saving graphics to use while working on remote desktop
            i +=1
            #self.patterns[func](self.xi,se)

            # plot the entire signal
            fig,ax = plt.subplots(subplot_kw={"projection":"3d"})
            surf = ax.plot_surface(self.X,self.T,np.real(self.f),cmap=cm.coolwarm,linewidth=0,antialiased=False)
            fig.colorbar(surf,shrink=0.5,aspect=5)
            plt.show()
            plt.savefig("figs/func_total.png")

        else:
            # runs the example for 2D (image) data; likely will use this in the presentation!!

            # need to make a movie or something here
            return 0
        
        return 0 # need to move

    def save_patterns(self):
        """
        
        Saves plots of f_i and f (composition of f_i) in desired file format (for presentations)
        
        """

        return 0 # need to move