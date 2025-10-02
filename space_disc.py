import numpy as np 
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy as sp
import scipy.linalg as sl
from scipy.fft import fft
from scipy.fft import ifft


#===========================================================================================
#===========================================================================================

####    definitions du problème

L = 10; Nx=200; h=1/Nx

Nx_2 = int((Nx/2)*(Nx%2==0) + ((Nx-1)/2)*(Nx%2==1))

I = np.linspace(-L/2,L/2,Nx,endpoint=False)
K = np.zeros(Nx)
K[:Nx_2] = np.arange(0,Nx_2); 

if(Nx%2==0): K[Nx_2:] = np.arange(-Nx_2,0) 
else:   K[Nx_2:] = np.arange(-Nx_2-1,0)

K2 = np.arange(0,Nx)



#===========================================================================================
#===========================================================================================

### Fonctions utiles 

def Fourier_coeffs(f):
    c = np.zeros(Nx, dtype='complex')
    for k in range(0,Nx_2):         #boucle sur les coeffs de fouriers
        for n in range(0,Nx):     #boucle sur la droite reel
            z= (-2j*np.pi*k*I[n]/L )
            c[k] += (f(I[n])*np.exp(z))
    
    for k in range(Nx_2,Nx):         #boucle sur les coeffs de fouriers
        for n in range(0,Nx):     #boucle sur la droite reel
            z= (-2j*np.pi*(k-Nx)*I[n]/L )
            c[k] += (f(I[n])*np.exp(z))
    return c


def Fourier_coeffs_non_opti(f):
    c = np.zeros(Nx, dtype='complex')
    for k in range(0,Nx):         #boucle sur les coeffs de fouriers
        for n in range(0,Nx):     #boucle sur la droite reel
            z= (-2j*np.pi*k*I[n]/L )
            c[k] += (f(I[n])*np.exp(z))
    return c

def derive_Fourier(f,D):
    c = np.zeros(Nx, dtype='complex')
    if(D==0):
        c= Fourier_coeffs(f)
    else: 
        
        for k in range(0,Nx_2):         #boucle sur les coeffs de fouriers
            
            for n in range(0,Nx):       #boucle sur la droite reel
                z= (2j*np.pi*k*I[n]/L)
                c[k] +=(f(I[n])*np.exp(z)) 
            
            for d in range(D):
                c[k] *=(2j*np.pi*k/L)
            
        for k in range(Nx_2,Nx):         #boucle sur les coeffs de fouriers
            for n in range(0,Nx):       #boucle sur la droite reel
                z= (2j*np.pi*(k-Nx)*I[n]/L)
                c[k] +=(f(I[n])*np.exp(z))
                
            for d in range(D):
                c[k] *=(2j*np.pi*(k-Nx)/L)
        return c
    return np.zeros(Nx)

def Inverse(c):
    P = np.zeros(Nx, dtype='complex')
    
    for k in range(0,Nx_2):
        for n in range(Nx):
            arg = (2j*np.pi*I[n]*k)/L 
            z = c[k]*np.exp(arg)
            P[n] += z
        
    for k in range(Nx_2,Nx):
        for n in range(Nx):
            arg = (2j*np.pi*I[n]*(k-Nx))/L 
            z = c[k-Nx]*np.exp(arg)
            P[n] += z
        
    return P*h

def Inverse_non_opti(c):
    P = np.zeros(Nx, dtype='complex')
    
    for k in range(0,Nx):
        for n in range(Nx):
            arg = (2j*np.pi*I[n]*k)/L 
            z = c[k]*np.exp(arg)
            P[n] += z
        
    return P*h



#===========================================================================================
#===========================================================================================


## fonctions tests
# F = lambda x: np.cos(2*np.pi*x/L)
# G = lambda x: -2*np.pi/L* np.sin(2*np.pi*x/L)
# H = lambda x: -2*np.pi/L * 2*np.pi/L* np.cos(2*np.pi*x/L)

# F = lambda x: x+2*x*x+3
# G = lambda x: 1+4*x
# H = lambda x: 4

# F = lambda x: (x<=0)*(1+2/L*x) + (x>0)*(1- 2/L*x)
# G = lambda x: 1*(x<0)- 1*(x>0)
# H = lambda x: -1*(x<h)*(x>-h) # approche un dirac lorsque Nx -> infty

# F = lambda x: 2*x**3 +3*x**2 +5*x+3
# G = lambda x: 6*x**2 +6*x +5
# H = lambda x: 12*x +6

F = lambda x: np.exp(np.cos(2*np.pi*x/L))
G = lambda x: -(2*np.pi/L) * np.exp(np.cos(2*np.pi*x/L))* np.sin(2*np.pi*x/L)
H = lambda x: (2*np.pi/L)**2 * np.exp(np.cos(2*np.pi*x/L))*(np.sin(2*np.pi*x/L)**2 - np.cos(2*np.pi*x/L))


#===========================================================================================
#===========================================================================================

### graphes 
C=Fourier_coeffs(F)
C2=Fourier_coeffs_non_opti(F)

IF  = Inverse(C)
IF2 = Inverse_non_opti(C2)

#===========================================================================================

P = fft(F(I))
PK,Pp = np.zeros(Nx,dtype='complex'), np.zeros(Nx,dtype='complex')
PPK,PPp = np.zeros(Nx,dtype='complex'), np.zeros(Nx,dtype='complex')


# sert de test de comparaison a la fonction derive_Fourier
# n utilise pas la periodicite
for k in range(0,Nx): 
    PK[k] =-(4j*np.pi*h/L)*k*P[k]
    PPK[k]=(4j*np.pi*h/L)*(4j*np.pi*h/L)*k *k*P[k]

Pp      = derive_Fourier(F,1)   #coeffs de la derive premier
PPp     = derive_Fourier(F, 2)  #coeffs de la derive seconde
IFp     = Inverse(Pp)           #derive premiere par fourier
IFFp    = Inverse(PPp)          #derive seconde par fourier


#===========================================================================================

Kinetic = -((2*np.pi/L)**2) *K*K                                          #correspond a la matrice Kinetic_energy du sujet
K_fft = (Kinetic * np.conjugate(sl.dft(Nx,'sqrtn')))@sl.dft(Nx,'sqrtn') #correspond a la matrice K du sujet

Kfp = np.real(K_fft @F(I))          #laplacien avec K
K_plot = ifft(Kinetic * fft(F(I)))  #laplacien avec Kinetic


#===========================================================================================

fig, (ax1,ax2,ax3) = plt.subplots(3, 1)

ax1.plot(I,F(I),'b')
ax2.plot(I,IF,'g')
ax3.plot(I,IF2,'r')
ax1.set_ylabel("fonction reel")
ax2.set_ylabel("fourier")
ax3.set_ylabel("fourier sans periodicite")
fig.suptitle("test fourier et periodicite")
plt.show()

#===========================================================================================

fig, (ax1,ax2,ax3) = plt.subplots(3, 1)

ax1.plot(I,F(I),'b')
ax2.plot(I,IF,'g')
ax3.plot(I,ifft(P),'r')
ax1.set_ylabel("fonction reel")
ax2.set_ylabel("inverse par fourier")
ax3.set_ylabel("inverse par fft")
fig.suptitle("test fourier et inverse")
plt.show()

#===========================================================================================

fig, (ax1,ax2,ax3) = plt.subplots(3, 1)

ax1.plot(I,G(I),'b')
ax2.plot(I,IFp,'g')
ax3.plot(I,ifft(PK),'r')
ax1.set_ylabel("fonction reel")
ax2.set_ylabel("derive par fourier")
ax3.set_ylabel("derive par fft")
fig.suptitle("test fourier et derive")
plt.show()

#===========================================================================================

fig, (ax1,ax2,ax3) = plt.subplots(3, 1)

ax1.plot(I,H(I),'b')
ax2.plot(I,IFFp,'g')
ax3.plot(I,ifft(PPK),'r')
ax1.set_ylabel("fonction reel")
ax2.set_ylabel("derive par fourier")
ax3.set_ylabel("derive par fft")
fig.suptitle("test fourier et laplacien")
plt.show()

#===========================================================================================

# fig, (ax1,ax2) = plt.subplots(2, 1)

# ax1.plot(np.imag(fft(F(I))),'b')
# ax2.plot(np.imag(C),'g')
# ax1.set_ylabel("coeff par fft")
# ax2.set_ylabel("coeff par fourier")
# fig.suptitle("coeff fourier imag")
# plt.show()

#===========================================================================================

# fig, (ax1,ax2) = plt.subplots(2, 1)

# ax1.plot(np.real(fft(F(I))),'b')
# ax2.plot(np.real(C),'g')
# ax1.set_ylabel("coeff par fft")
# ax2.set_ylabel("coeff par fourier")
# fig.suptitle("coeff fourier real")
# plt.show()

#===========================================================================================

# plt.plot(I,K_plot,'b')
# plt.plot(I,Kfp,'k')
# plt.plot(I,H(I),'g')
# plt.title("test fft et array")
# plt.show()

#===========================================================================================
#===========================================================================================
