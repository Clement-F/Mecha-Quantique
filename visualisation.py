import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np 
import numpy.random as npr
import scipy as sp
import scipy.linalg as sl
from scipy.fft import fft
from scipy.fft import ifft


# Simulates the SchrÃ¶dinger dynamics iâˆ‚t = -1/2 Ïˆ'' + V(x,t) Ïˆ, with the pseudospectral method
# on an interval [-L,L] with periodic boundary conditions, with Nx grid points
# The simulation proceeds from 0 to T, with Nt time steps.
# Initial condition is given by the function psi0_fun(x), potential by the function V_fun(x,t)
# Returns an array psi[ix, it]


def dynamics_Kfft(psi0_fun=(lambda x: np.exp(-x**2)), V_fun=(lambda x,t: 0), L=10, Nx=100, T=4, Nt=100):

    Nx_2 = int((Nx/2)*(Nx%2==0) + ((Nx-1)/2)*(Nx%2==1)) 

    K = np.zeros(Nx)
    K[:Nx_2] = np.arange(0,Nx_2); 

    if(Nx%2==0): K[Nx_2:] = np.arange(-Nx_2,0) 
    else:   K[Nx_2:] = np.arange(-Nx_2-1,0)

<<<<<<< HEAD
    Kinetic = 0.5*(2*np.pi/L)**2 *K*K
    K_fft = (Kinetic * np.conjugate(sl.dft(Nx,'sqrtn')))@sl.dft(Nx,'sqrtn')
=======
    Kinetic = -(2*np.pi/L)**2 *K*K
    K_fft =  -0.5 *(Kinetic *sl.dft(Nx,'sqrtn'))@sl.dft(Nx,'sqrtn')
>>>>>>> 12f5e00750bad071987c8c976428f8d901d1a78c

    I = np.linspace(-L, L,Nx)
    Psi_0T = np.zeros((Nx,Nt), dtype="complex")
    dt = T/Nt
    K_dt = -1j*K_fft*dt
    
    # evo_dt  = sl.expm(K_dt)
    Psi_0T[:,0]=psi0_fun(I)
    
    for i in range(1,Nt):
        ti =    dt*i

        Vti =    -1j*V_fun(I,ti)*ti;    K_ti =  -1j*K_fft*ti
        eVti =   np.exp(Vti)      ;    eKti = sl.expm(K_ti)
        Psi_0T[:,i]=eKti@( eVti * Psi_0T[:,i-1])
        print(i,ti)
    return Psi_0T




def dynamics_fft(psi0_fun=(lambda x: np.exp(-x**2)), V_fun=(lambda x,t: 0), L=10, Nx=100, T=4, Nt=100):

    Nx_2 = int((Nx/2)*(Nx%2==0) + ((Nx-1)/2)*(Nx%2==1)) 

    K = np.zeros(Nx)
    K[:Nx_2] = np.arange(0,Nx_2); 

    if(Nx%2==0): K[Nx_2:] = np.arange(-Nx_2,0) 
    else:   K[Nx_2:] = np.arange(-Nx_2-1,0)

    Kinetic = 0.5*(2*np.pi/L)**2 *K*K
    K_fft = (Kinetic * np.conjugate(sl.dft(Nx,'sqrtn')))@sl.dft(Nx,'sqrtn')

    I = np.linspace(-L, L,Nx)
    Psi_0T = np.zeros((Nx,Nt), dtype="complex")
    dt = T/Nt
    K_dt = -1j*K_fft*dt
    
    evo_t  = np.eye(Nt)
    Psi_0T[:,0]=psi0_fun(I)
    
    for i in range(1,Nt):
        ti = dt*i
        Psi_0T[:,i] = ifft(-1j*Kinetic *ti * fft(Psi_0T[:,0]))
        print(ti,np.sqrt(L/Nx)*np.linalg.norm(Psi_0T[:,i]))
    return Psi_0T

# Plots the return value psi of the function "dynamics", using linear interpolation
# The whole of psi is plotted, in an animation lasting "duration" seconds (duration is unconnected to T)
# L argument is only for x axis labelling


def plot_psi(psi, duration=10, frames_per_second=30, L=10):
    
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
    proba_plot = ax.plot(x_data, np.abs(psi[:,0]**2),label ='proba')[0]
    V_plot = ax.plot(x_data, V(x_data,0), label='potentiel')[0]
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
        proba_plot.set_ydata((np.abs(psi_t)**2))
        V_plot.set_ydata(V(x_data,t))

    ani = animation.FuncAnimation(fig=fig, func=update, frames=duration*frames_per_second, interval=1000/frames_per_second)
    return ani

<<<<<<< HEAD
L=10; Nx=100; Nt=100; T=1
V = lambda x,t : 10000*(x<=-2*L/4) 
J = np.linspace(-L,L,Nx)
psi = dynamics_fft(psi0_fun=psi0,L=L,Nx=Nx,T=T, Nt=Nt)
=======
a=1
psi0=(lambda x: np.exp(-a*x*x))

L=10; Nx=200; Nt=200; T=5

V = lambda x,t : 1000*(x>L/2) + 1000*(x<-L/2) -10*(x<=L/2)*(x>=-L/2) # puit d energie
# V = lambda x,t :10000*(x<=3*L/4)*(x>=L/4)
J = np.linspace(-L,L,Nx)
psi = dynamics(psi0_fun=psi0, V_fun=V, L=L, Nx=Nx, T=T, Nt=Nt)
>>>>>>> 12f5e00750bad071987c8c976428f8d901d1a78c


# plt.plot(J,V(J,0))
# plt.plot(J,np.abs(psi[:,-1])**2)
# plt.plot(J,np.abs(psi[:,1])**2)
# plt.show()


anime = plot_psi(psi,L=L, duration=20, frames_per_second=60)
plt.show()
