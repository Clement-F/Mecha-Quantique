import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np 
import numpy.random as npr
import scipy as sp
import scipy.linalg as sl



# def Fourier_coeffs(f):
#     c = np.zeros(Nx, dtype='complex')
#     for k in range(Nx):         #boucle sur les coeffs de fouriers
#         for n in range(Nx):     #boucle sur la droite reel
#             z= -(2j*np.pi*k*I[n]/L )
#             c[k] += (f(I[n])*np.exp(z))
#     return c



# def derive_Fourier(f,D):
#     c = np.zeros(Nx, dtype='complex')
#     if(D==0):
#         return Fourier_coeffs(f)
#     else: 
#         for k in range(Nx):         #boucle sur les coeffs de fouriers
#             for n in range(Nx):     #boucle sur la droite reel
#                 z= -(2j*np.pi*k*I[n]/L)
#                 c[k] +=(f(I[n])*np.exp(z))   
#             for d in range(D):
#                 c[k] *=-(2j*np.pi*k/L)*h
#     return 2*c


# def Inverse(c):
#     P = np.zeros(Nx, dtype='complex')
#     for k in range(0,Nx):
#         arg = (2j*np.pi*I*k)/L 
#         z = c[k]*np.exp(arg)
#         P += z*h
#     return np.real(P)


# Kinetic = -2*np.pi/L*2*np.pi/L*sl.dft(Nx,'n') @ sl.dft(Nx)


# Simulates the SchrÃ¶dinger dynamics iâˆ‚t = -1/2 Ïˆ'' + V(x,t) Ïˆ, with the pseudospectral method
# on an interval [-L,L] with periodic boundary conditions, with Nx grid points
# The simulation proceeds from 0 to T, with Nt time steps.
# Initial condition is given by the function psi0_fun(x), potential by the function V_fun(x,t)
# Returns an array psi[ix, it]


def dynamics(psi0_fun=(lambda x: np.exp(-x**2)), V_fun=(lambda x,t: 0), L=10, Nx=200, T=4, Nt=200):

    Kinetic = -  np.pi/L**2 *sl.dft(Nx,'n') @ sl.dft(Nx)
    I = np.linspace(-L, L,Nx)
    Psi_0T = np.zeros((Nx,Nt), dtype="complex")
    dt = T/Nt
    K_dt = -1j*Kinetic*dt
    evo_dt = sl.expm(K_dt) 
    
    evo_t  = np.eye(Nt)
    
    for i in range(1,Nt):
        ti = dt*i
        evo_t = evo_t @ evo_dt
        Psi_0T[:,i]=evo_t @ psi0_fun(I)
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


psi = dynamics()
plt.plot(psi[:,-1])
plt.plot(psi[:,1])
plt.show()
#plot_psi(psi)
