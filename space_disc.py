import numpy as np 
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy as sp
import scipy.linalg as sl


####    definitions du problème

L=2*np.pi; Nx=200; h=1/Nx
I = np.linspace(-L/2,L/2,Nx,endpoint=False)
K = np.arange(Nx)

### Fonctions utiles 

def Fourier_coeffs(f):
    c = np.zeros(Nx, dtype='complex')
    for k in range(Nx):         #boucle sur les coeffs de fouriers
        for n in range(Nx):     #boucle sur la droite reel
            z= -(2j*np.pi*k*I[n]/L )
            c[k] += (f(I[n])*np.exp(z))
    return c



def derive_Fourier(f,D):
    c = np.zeros(Nx, dtype='complex')
    if(D==0):
        return Fourier_coeffs(f)
    else: 
        for k in range(Nx):         #boucle sur les coeffs de fouriers
            for n in range(Nx):     #boucle sur la droite reel
                z= -(2j*np.pi*k*I[n]/L)
                c[k] +=(f(I[n])*np.exp(z))   
            for d in range(D):
                c[k] *=-(2j*np.pi*k/L)*h
    return 2*c


def Inverse(c):
    P = np.zeros(Nx, dtype='complex')
    for k in range(0,Nx):
        arg = (2j*np.pi*I*k)/L 
        z = c[k]*np.exp(arg)
        P += z*h
    return np.real(P)


Kinetic = -2*np.pi/L*2*np.pi/L*sl.dft(Nx,'n') @ sl.dft(Nx)

#%%

## fonctions tests
F = lambda x: np.cos(2*np.pi*x/L)
G = lambda x: -np.sin(x)
H = lambda x: -2*np.pi/L * 2*np.pi/L* np.cos(2*np.pi*x/L)
#F = lambda x: x+2*x*x+3
#G = lambda x: 1+4*x
#F = lambda x: np.exp(np.cos(2*np.pi*x/L))


### graphes 

C=Fourier_coeffs(F)
P = np.fft.fft(F(I))

Pp = derive_Fourier(F,2)
IFp = Inverse(Pp)

Kfp = Kinetic@F(I)


plt.plot(I,IFp,'b')
plt.plot(I,Kfp,'k')
plt.plot(I,H(I),'g')
plt.show()


# fig, (ax1,ax2) = plt.subplots(2, 1)

# ax1.plot(I,F(I),'b')
# ax2.plot(I,IF,'g')
# plt.show()


# fig, (ax1,ax2) = plt.subplots(2, 1)

# ax1.plot(np.imag(P),'r')
# ax2.plot(np.imag(C),'g')
# plt.show()


# fig, (ax1,ax2) = plt.subplots(2, 1)

# ax1.plot(np.real(P),'r')
# ax2.plot(np.real(C),'g')
# plt.show()
#%%
Nt=100; Nx=100; T=4
psi0_fun = lambda x: np.exp(-x*x)
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
