import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np 
import numpy.random as npr
import scipy as sp
import scipy.linalg as sl
from scipy.fft import fft
from scipy.fft import ifft
from scipy.fft import fft2
from scipy.fft import ifft2
import time

from mpl_toolkits.mplot3d import Axes3D
# ============================================================================================
# ============================================================================================
# ============================================================================================


def dynamics_1D(psi0_fun=(lambda x: np.exp(-x**2)), V_fun=(lambda x,t: 0), L=10, Nx=100, T=4, Nt=100):

    Nx_2 = int((Nx/2)*(Nx%2==0) + ((Nx-1)/2)*(Nx%2==1)) 

    K = np.zeros(Nx)
    K[:Nx_2] = np.arange(0,Nx_2); 

    if(Nx%2==0): K[Nx_2:] = np.arange(-Nx_2,0) 
    else:   K[Nx_2:] = np.arange(-Nx_2-1,0)

    Kinetic = 0.5*(2*np.pi/L)**2 *K*K
    I = np.linspace(-L, L,Nx)
    Psi_0T = np.zeros((Nx,Nt), dtype="complex")
    dt = T/Nt

    Psi_0T[:,0]=psi0_fun(I)
    
    for i in range(1,Nt):
        ti = dt*i
        Psi_0T[:,i] = ifft((np.exp(-1j*Kinetic*dt)) * fft(np.exp(-1j*V_fun(I,ti)*ti) * Psi_0T[:,i-1]))
        print(ti,np.sqrt(L/Nx)*np.linalg.norm(Psi_0T[:,i]))
    return Psi_0T

# ============================================================================================
# ============================================================================================


def plot_psi_1D(psi, duration=10, frames_per_second=30, L=10):
    
    fig, ax = plt.subplots()
    t_data = np.linspace(0, 1, np.size(psi, 1)) # 1 is arbitrary here
    x_data = np.linspace(-L,L,np.size(psi,0), endpoint=False)
    # set the min and maximum values of the plot, to scale the axis
    m = min(0, np.min(np.real(psi)), np.min(np.imag(psi)))
    M = np.max(np.abs(psi))
    
    # set the axis once and for all
    ax.set(xlim=[-L,L], ylim=[m,M], xlabel='x', ylabel='psi')
    
    # dummy plots, to update during the animation
    real_plot = ax.plot(x_data, np.real(psi[:, 0]), label='Real')[0]
    imag_plot = ax.plot(x_data, np.imag(psi[:, 0]), label='Imag')[0]
    abs_plot  = ax.plot(x_data, np.abs(psi[:, 0]), label='Abs')[0]
    ax.legend()

    # define update function as an internal function (that can access the variables defined before)
    # will be called with frame=0...(duration*frames_per_second)-1
    def update(frame):
        print(frame)
        # get the data by linear interpolation
        t = frame / (duration * frames_per_second)
        psi_t = np.array([np.interp(t, t_data, psi[i, :]) for i in range(np.size(psi,0))])
        # update the plots
        real_plot.set_ydata(np.real(psi_t))
        imag_plot.set_ydata(np.imag(psi_t))
        abs_plot.set_ydata(np.abs(psi_t))

    ani = animation.FuncAnimation(fig=fig, func=update, frames=duration*frames_per_second, interval=1000/frames_per_second)
    return ani


def plot_psi_2D(psi, duration=10, frames_per_second=30, Lx=10, Ly=10):
    
    fig, ax = plt.subplots()
    

    ims=[]
    for i in range(60):
        im = ax.imshow(np.abs(psi[:,:,i]),extent=[-Lx, Lx, -Ly, Ly], animated=True)
        if i == 0:
            ax.imshow(np.abs(psi[:,:,i]),extent=[-Lx, Lx, -Ly, Ly] )
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)
    return ani


# =============================================================================================
# =============================================================================================

savefile = 5
red_x = 2 ;red_y=2

def Norme (Nx,Ny,Lx,Ly,psi):
    return np.sqrt(Lx/Nx)*np.sqrt(Ly/Ny)*np.linalg.norm(psi)

def Projection_red (Nx,Ny,psi,red_x=red_x,red_y=red_y):
    sol_red = np.zeros((int(Nx/red_x),int(Ny/red_y)), dtype="complex")
    for i in range(0,Nx,red_x):
        for j in range(0,Ny,red_y):
            sol_red[int(i/red_x),int(j/red_y)] = psi[i,j]
    return sol_red

# =============================================================================================
# =============================================================================================


def Kin_2D(Nx,Ny):
    Nx_2 = int((Nx/2)*(Nx%2==0) + ((Nx-1)/2)*(Nx%2==1)) 
    Ny_2 = int((Ny/2)*(Ny%2==0) + ((Ny-1)/2)*(Ny%2==1))

    K = np.zeros((Nx,Ny))
    for i in range(0,Nx_2):
        for j in range(0,Ny_2):
            K[i,j] = 0.5*(2*np.pi/L)**2 *i*j
            
        for j in range(Ny_2,Ny-1):
            K[i,j] = 0.5*(2*np.pi/L)**2 *i*(j-Ny)
    
    for i in range(Nx_2,Nx-1):
        for j in range(0,Ny_2):
            K[i,j] = 0.5*(2*np.pi/L)**2 *(i-Nx)*j
            
        for j in range(Ny_2,Ny-1):
            K[i,j] = 0.5*(2*np.pi/L)**2 *(i-Nx)*(j-Ny)

    return K


def dynamics_2D(psi0_fun=(lambda x,y: np.exp(-(x*x+y*y)**2)), V_fun=(lambda x,y,t: 0), Lx=10, Ly=10, Nx=100, Ny=100, T=4, Nt=100):


    Kinetic = Kin_2D(Nx,Ny)
    I = np.linspace(-Lx,Lx,Nx); J = np.linspace(-Ly,Ly,Ny).reshape(-1,1)
    Psi_T = np.zeros((int(Nx/red_x),int(Ny/red_y),Nt), dtype="complex")
    Psi_temp =np.zeros((Nx,Ny,2), dtype="complex")
    dt = T/Nt

    Psi_temp[:,:,0]=psi0_fun(I,J); Psi_T[:,:,0]= Projection_red(Nx,Ny,Psi_temp[:,:,0])
    norm = np.sqrt(Lx/Nx)*np.sqrt(Ly/Ny)*np.linalg.norm(Psi_temp[:,:,0])
    
    diff_Norm = np.zeros(Nt); Energie = np.zeros(Nt)

    for i in range(1,savefile*Nt):
        ti = dt*i
        Psi_temp[:,:,1]= ifft2((np.exp(-1j*Kinetic*dt)) * fft2(np.exp(-1j*V_fun(I,J,ti)*ti) *Psi_temp[:,:,0]))
        if(i%savefile==0) : 
            Psi_T[:,:,int(i/savefile)]  = Projection_red(Nx,Ny,Psi_temp[:,:,1]); 

            diff_Norm[int(i/savefile)]  = np.log(np.abs(norm -Norme(Nx,Ny,Lx,Ly, Psi_temp[:,:,1])))
            Energie[int(i/savefile)]    = np.sqrt(Lx/Nx)*np.sqrt(Ly/Ny)*np.linalg.norm(ifft2( Kinetic * fft2(V_fun(I,J,ti) *Psi_temp[:,:,1])))

            print(i,diff_Norm[int(i/savefile)], Energie[int(i/savefile)])
        Psi_temp[:,:,0] = Psi_temp[:,:,1]

    return Psi_T, diff_Norm, Energie


# ============================================================================================

r=1
a=r*np.sqrt(2*np.log(2)); kx=0; ky=2; x0=1; y0=0

N=400;      L=10
Lx=L;   Nx=N;   Ly=L;   Ny=N
L = max(Lx,Ly)
Nt=2500;     T=25
l=np.sqrt(25);      V0=1
psi0= lambda x,y: np.exp(-a*(x*x + y*y)) *np.exp(1j*kx*(x-x0)) *np.exp(1j*ky*(y-y0))
# psi0 = lambda x,y : 2/(np.sqrt(2*np.pi*a*a)- np.sqrt(np.pi*a*a))* np.exp(-(x*x + y*y)/(2*a*a))*(1-np.exp(-(x*x + y*y)/(2*a*a)))*np.exp(1j*kx*(x-x0)) *np.exp(1j*ky*(y-y0))       # cercle autour de l'origine
# psi0= lambda x: np.exp(-a*(x-x0)*(x-x0)) *np.exp(1j*k*(x-x0))




# V = lambda x,y,t : V0*((x*x + y*y)>(l)**2) -V0*((x*x + y*y)<=(l)**2)*((x*x + y*y)>=-(l)**2)     # puit d energie
# V = lambda x,y,t : V0*((x*x + y*y)<=(L/2+l)**2)*((x*x+y*y)>=(L/2)**2)  # barriere (effet tunnel)
V = lambda x,y,t : -5*V0/np.abs(x*x + y*y) 
# V = lambda x,y,t : 1*np.cos(2*np.pi*x/L)
V = lambda x,y,t : 0 * x * y


I = np.linspace(-Lx,Lx,Nx); J = np.linspace(-Ly,Ly,Ny).reshape(-1,1); Time = np.arange(0,Nt)

fig, ax = plt.subplots()
im = ax.imshow(V(I,J,0),extent=[-Lx, Lx, -Ly, Ly], animated=True)

psi, diff_norm, Ener = dynamics_2D(psi0_fun=psi0,V_fun=V, Lx=Lx,Ly=Ly, Nx=Nx, Ny=Ny, T=T, Nt=Nt)
anime = plot_psi_2D(psi,Lx=Lx, Ly=Ly, duration=2*T, frames_per_second=60)

plt.show()
plt.plot(Time, diff_norm)
plt.plot(Time, Ener)
plt.show()