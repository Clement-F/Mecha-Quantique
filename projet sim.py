import numpy as np 
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy as sp


####    definitions du problème

L=1; Nx=100; h=1/Nx 
I = np.linspace(-L/2,L/2,Nx,endpoint=False)
K = np.arange(Nx)





def fourier_coeffs(f):
    cr = np.zeros(Nx); ci = np.zeros(Nx)
    for k in range(Nx):         #boucle sur les coeffs de fouriers
        for n in range(Nx):     #boucle sur la droite reel
            z= -(2*np.pi*k*I[n]/L )*1j
            cr[k] += (f(I[n])*np.exp(z)).real
            ci[k] += (f(I[n])*np.exp(z)).imag
    cr*=-h/2; ci*=h/2;
    return cr,ci

def Inverse(cr,ci):
    Pr = np.zeros(Nx); Pi = np.zeros(Nx)
    for k in range(Nx):
        print(k)
        arg = 2*np.pi*I*k/L *1j
        ck = cr[k]+1j*ci[k]
        z1 = ck*np.exp(arg)
        z2 = ck*np.exp(-arg)
        # arg = 2*np.pi*I*(k+Nx)/L *1j
        # z3 = (cr[k]+1j*ci[k])*np.exp(arg)
        # z4 = (cr[k]+1j*ci[k])*np.exp(-arg)
        z = z1+z2
        Pr += z.real; Pi += z.imag
    return Pr,Pi



F = lambda x: np.sin(2*np.pi*x/L)
#F = lambda x: x+2*x*x+3

Cr,Ci=fourier_coeffs(F)
P = np.fft.fft(F(I))

Pulse = Cr + 1j*Ci

#plt.plot(I,F(I),'b')
plt.plot(I,Inverse(Cr,Ci)[0],'g')
plt.plot(I,np.fft.ifft(Pulse),'k')
#plt.plot(I,np.fft.ifft(P),'r')
plt.show()

freq = np.fft.fftfreq(I.shape[-1])

# plt.plot(np.imag(P))
# plt.plot(Ci)
# plt.show()


# plt.plot(np.real(P))
# plt.plot(Cr)
# plt.show()



fig, (ax1,ax2) = plt.subplots(2, 1)

ax1.plot(np.imag(P),'r')
ax2.plot(Ci,'g')
plt.show()


fig, (ax1,ax2) = plt.subplots(2, 1)

ax1.plot(np.real(P),'r')
ax2.plot(Cr,'g')
plt.show()
