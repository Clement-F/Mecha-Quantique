import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np 
import numpy.random as npr
import scipy as sp
import scipy.linalg as sl
from scipy.fft import fft
from scipy.fft import ifft


# ============================================================================================
# ============================================================================================

splin_exp = lambda x: np.exp(-1/x)*(x>0) / (np.exp(-1/x)*(x>0)+ np.exp(-1/(1-x))*(x>0))
raccord = lambda x,p1,p2,a,b: (1-splin_exp((x-a)/(b-a)))*p1 + splin_exp((x-a)/(b-a))*p2 

def barriere_reg(X,t,x1,x2,epsi):
    V=np.zeros(X.size)

    for i in range(X.size) :
        x = X[i]

        if(x<x1-epsi):                      V[i] = 0 
        elif((x1-epsi<x)  and (x<x1+epsi)): V[i] = raccord(x,0,V0,x1-epsi,x1+epsi)
        elif((x1+epsi<x)  and (x<x2-epsi)): V[i] = V0
        elif((x2-epsi<x)  and (x<x2+epsi)): V[i] = raccord(x,V0,0,x2-epsi,x2+epsi)
        elif(x2+epsi<x):                    V[i] = 0

    return V


def Kin (Nx):
    
    Nx_2 = int((Nx/2)*(Nx%2==0) + ((Nx-1)/2)*(Nx%2==1)) +1
    K = np.zeros((Nx), dtype='complex')
    for i in range(0,Nx_2-1):
        K[i] = 0.5*(2*np.pi/L)**2 *i*i
        print(i)
    for i in range(Nx_2,Nx-1):
        K[i] = 0.5*(2*np.pi/L)**2 *(i-Nx)*(i-Nx)
        print(i)
    return K

def Concentration_de_masse(psi,Nx,L,a,b):
    l = np.abs(b-a);    occupation  = l/(2*L) 
    an = int((np.abs(a+L)/(2*L))*Nx) ; bn = int((np.abs(b+L)/(2*L))*Nx)
    N = np.abs(bn-an)
    print(l,N,an,bn)
    psi_l = psi[an:bn]
    masse = np.sqrt(l/N)*np.linalg.norm(psi_l)
    return masse



# ============================================================================================
# ============================================================================================

# Simulates the SchrÃ¶dinger dynamics iâˆ‚t = -1/2 Ïˆ'' + V(x,t) Ïˆ, with the pseudospectral method
# on an interval [-L,L] with periodic boundary conditions, with Nx grid points
# The simulation proceeds from 0 to T, with Nt time steps.
# Initial condition is given by the function psi0_fun(x), potential by the function V_fun(x,t)
# Returns an array psi[ix, it]


def dynamics_Kfft(psi0_fun=(lambda x: np.exp(-x**2)), V_fun=(lambda x,t: 0), L=10, Nx=1000, T=4, Nt=1000):

    #declaration de variable d interet
    Nx_2 = int((Nx/2)*(Nx%2==0) + ((Nx-1)/2)*(Nx%2==1)) 
    K = np.zeros(Nx); K[:Nx_2] = np.arange(0,Nx_2); 

    if(Nx%2==0): K[Nx_2:] = np.arange(-Nx_2,0) 
    else:   K[Nx_2:] = np.arange(-Nx_2-1,0)

    Kinetic = 0.5*(2*np.pi/L)**2 *K*K
    K_fft = (Kinetic * np.conjugate(sl.dft(Nx,'sqrtn')))@sl.dft(Nx,'sqrtn')

    I = np.linspace(-L, L,Nx,endpoint=False)
    Psi_T = np.zeros((Nx,Nt), dtype="complex")
    dt = T/Nt

    # K_dt = -1j*K_fft*dt
    # evo_dt  = sl.expm(K_dt)

    Psi_T[:,0]=psi0_fun(I)
    
    for i in range(1,Nt):
        ti =    dt*i
        Vti =    -1j*V_fun(I,ti)*ti;    K_ti =  -1j*K_fft*ti
        eVti =   np.exp(Vti)      ;    eKti = sl.expm(K_ti)
        Psi_T[:,i]=eKti@( eVti * Psi_T[:,0])
        print(i,ti)
    return Psi_T

def dynamics_fft(psi0_fun=(lambda x: np.exp(-x**2)), V_fun=(lambda x,t: 0), L=10, Nx=100, T=4, Nt=100):

    Nx_2 = int((Nx/2)*(Nx%2==0) + ((Nx-1)/2)*(Nx%2==1)) 

    K = np.zeros(Nx)
    K[:Nx_2] = np.arange(0,Nx_2); 

    if(Nx%2==0): K[Nx_2:] = np.arange(-Nx_2,0) 
    else:   K[Nx_2:] = np.arange(-Nx_2-1,0)

    Kinetic = 0.5*(2*np.pi/L)**2 *K*K
    I = np.linspace(-L, L,Nx,endpoint=False)
    Psi_T = np.zeros((Nx,Nt), dtype="complex")
    dt = T/Nt

    Psi_T[:,0]=psi0_fun(I)
    
    for i in range(1,Nt):
        ti = dt*i
        Psi_T[:,i] = ifft((np.exp(-1j*Kinetic*ti)) * fft(np.exp(-1j*V_fun(I,ti)*ti) * Psi_T[:,0]))
        print(ti,np.sqrt(L/Nx)*np.linalg.norm(Psi_T[:,i]))
    return Psi_T

def dynamics_fft_diss(psi0_fun=(lambda x: np.exp(-x**2)), V_fun=(lambda x,t: 0), L=10, Nx=100, T=4, Nt=100):

    Nx_2 = int((Nx/2)*(Nx%2==0) + ((Nx-1)/2)*(Nx%2==1)) 

    # K = np.zeros(Nx)
    # K[:Nx_2] = np.arange(0,Nx_2); 

    # if(Nx%2==0): K[Nx_2:] = np.arange(-Nx_2,0) 
    # else:   K[Nx_2:] = np.arange(-Nx_2,1)

    dt = T/Nt; dx = L/Nx
    # Kinetic = (0.5*(2*np.pi/L)**2) *K*K
    Kinetic = (0.5*(2*np.pi/L)**2) *np.fft.fftfreq(Nx, dx)*np.fft.fftfreq(Nx, dx)
    I = np.linspace(-L, L,Nx,endpoint=False)
    Psi_T = np.zeros((Nx,Nt), dtype="complex")
    Phi_T = np.zeros((Nx,Nt), dtype="complex")

    Psi_T[:,0]=psi0_fun(I)
    
    
    for i in range(1,Nt):
        ti = dt*i
        Phi_T[:,i] = (np.exp(-1j*Kinetic*dt)) * fft(np.exp(-1j*V_fun(I,ti)*dt) * Psi_T[:,i-1])
        Psi_T[:,i] = ifft(Phi_T[:,i])
        # print(ti,np.sqrt(L/Nx)*np.linalg.norm(Psi_T[:,i]))
    return Psi_T, Phi_T


# ============================================================================================
# ============================================================================================


# Plots the return value psi of the function "dynamics", using linear interpolation
# The whole of psi is plotted, in an animation lasting "duration" seconds (duration is unconnected to T)
# L argument is only for x axis labelling


def plot_psi(psi, duration=10, frames_per_second=30, L=10, show_potential=False):
    
    fig, ax = plt.subplots()
    t_data = np.linspace(0, 1, np.size(psi, 1)) # 1 is arbitrary here
    x_data = np.linspace(-L,L,  np.size(psi,0))
    # set the min and maximum values of the plot, to scale the axis
    m = min(0, np.min(np.real(psi)), np.min(np.imag(psi)))
    M = np.max(np.abs(psi))
    
    # set the axis once and for all
    ax.set(xlim=[-L,L], ylim=[m,M], xlabel='x', ylabel='psi')
    
    # dummy plots, to update during the animation
    real_plot = ax.plot(x_data, np.real(psi[:, 0]), label='Real')[0]
    imag_plot = ax.plot(x_data, np.imag(psi[:, 0]), label='Imag')[0]
    abs_plot  = ax.plot(x_data, np.abs(psi[:, 0]), label='Abs')[0]
    if(show_potential):V_plot  =   ax.plot(x_data, V(x_data,0), label='V')[0]
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



# ============================================================================================
# ============================================================================================

# r=2
# a=r*np.sqrt(2*np.log(2)); kx=2; x0=1

a=1; kx=20; x0=0
psi0= lambda x: 1/(np.sqrt(2*np.pi *a*a)) *np.exp(-(x*x)/(2*a*a)) *np.exp(1j*kx*(x-x0))
# psi0= lambda x: 2/(np.sqrt(2*np.pi*a*a)- np.sqrt(np.pi*a*a))* np.exp(-(x*x)/(2*a*a))*(1-np.exp(-(x*x)/(2*a*a)))*np.exp(1j*kx*(x-x0))       # cercle autour de l'origine
# psi0 = lambda x: np.exp(-1j*kx*x) 

L=10; Nx=1000; Nt=4000; T=100
l=1; s=2; V0=100; epsi=0.1



an = int((np.abs(s+L)/(2*L))*Nx) ; bn = int((np.abs(L+L)/(2*L))*Nx)
print(an,bn)

# V = lambda x,t : V0*(x>L/2) + V0*(x<-L/2)                           # puit d energie
# V = lambda x,t : V0*(x<=s+l )*(x>=s)                                # barriere (effet tunnel)
V = lambda x,t : barriere_reg(x,t,s,s+l,epsi) + barriere_reg(x,t,-s-l,-s,epsi)                             # barriere (effet tunnel)
# V = lambda x,t : 1/np.abs(x)
# V = lambda x,t : 1*np.cos(2*np.pi*x/L)
# V = lambda x,t : V0*x*x
# V = lambda x,t : 0 * x

# plt.plot(J,V(J,0))
# plt.show()


# =====================================================

def Jauges(psi,Nx,L,t):
    dx = 1/Nx
    I = np.linspace(-L,L,Nx); Kinetic = (0.5*(2*np.pi/L)**2) *np.fft.fftfreq(Nx, dx)*np.fft.fftfreq(Nx, dx)
    print("norme de psi sur [-L,L] = ", np.sqrt(2*L/Nx)*np.linalg.norm(psi))
    print("norme de psi en dehors du puit = ", Concentration_de_masse(psi=psi,Nx=Nx,L=L,a=s,b=L)+Concentration_de_masse(psi=psi,Nx=Nx,L=L,a=-s,b=-L) )
    print("Energie de psi = ",np.sqrt(2*L/Nx)*np.linalg.norm(np.abs( ifft( Kinetic * fft(V(I,t) *psi)))) )
    print("============================== ",t," ===========================")

# =====================================================


psi,phi = dynamics_fft_diss(psi0_fun=psi0,V_fun=V, L=L, Nx=Nx, T=T, Nt=Nt)
J =np.linspace(-L,L,Nx, endpoint=False)
dt = 1/Nt
for k in range(0,Nt,10):
    Jauges(psi[:,k],Nx,L,dt*k)

# P = psi0(J)
# dx =1/Nx; dt =1/Nt
# Kinetic = (0.5*(2*np.pi/L)**2) *np.fft.fftfreq(Nx, Nx)*np.fft.fftfreq(Nx, Nx)
# ti =dt
# Phi_T = (np.exp(-1j*Kinetic*dt)) * fft(P)
# Psi_T = ifft(Phi_T)
# print(P)
# print("==========================")
# print(Psi_T)
# print("==========================")
# print(Phi_T)
# plt.plot(J,np.imag(P))
# plt.plot(J,np.real(P))
# plt.show()


# anime_phi = plot_psi(phi,L=L, duration=10, frames_per_second=20)
anime_psi = plot_psi(psi,L=L, duration=10, frames_per_second=20,show_potential=True)
plt.show()

