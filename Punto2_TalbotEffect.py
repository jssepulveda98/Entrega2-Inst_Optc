"""
Fresnel Transform using FFT

ESQUEMA
1. Calculate U'[n,m,0] multiplying by a phase
2. Calculate U''[n,m,z] using FFT
3. Calculate U[n,m,z] multiplying by a scaling function
4. Reordenar el campo
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

def transmittance(dx0, N, m, l):
    """
    Incident wave and transmittance function
    In this case: monochromatic plane wave and diffraction grating
    """
    dy0=dx0
    x=np.arange(-N/2,N/2)
    y=np.arange(-N/2,N/2)
    x,y=np.meshgrid(x,y)
    t=(1/2)*(1+m*np.cos((2*np.pi*x*dx0)/l))

    return t

def Fresnel(Uin, w_l, dx0, dx, z):
    "-----Step 1------"
    k=2*np.pi/w_l
    N,M=np.shape(Uin)
    x=np.arange(-N/2,N/2,1)
    y=np.arange(-M/2,M/2,1)
    X,Y=np.meshgrid(x,y)
    phase=np.exp((1j*k)/(2*z)*(((X*dx0)**2) + ((Y*dx0)**2)))
    U1=Uin*phase
    "-----Step 2-----"
    #X=X*(1/(M*dx0))
    #Y=Y*(1/(N*dx0))
    Uf=np.fft.fftshift(np.fft.fft2(U1*dx0**2))
    "-----Step 3-----"

    c1=np.exp(1j*k*z)/(1j*w_l*z)
    Uf=Uf*c1*(np.exp((1j*(k/2*z))*((X*dx)**2 + (Y*dx)**2)))
    
    return Uf

"-----Physical array-----"

"""
U=incident wave
z=entrance plane to detector plane distance
w_l= wavelength
dx=dy=pixel size detector plane
dx0=dy0=pixel size entrance plane
M=number of pixels in the x axis
N=number of pixels in the y axis
(M=N)
MxN=number of pixels
"""
        
w_l=0.633          #(633nm orange/red)   #All units in um
dx0=2.5            #2.5um
N=M=2048           #Number of pixels
z=20500            #20.5 mm
#Diffraction grating parameters
m=1000                #Contrast factor
l=512              #Period


tic=time.time()


dx=w_l*z/(dx0*N)
print ("dx:",dx)
lim=N*(dx0**2)/w_l  #Limit of z in FT
print ("lim:",lim)
if z<lim:
    print("z limit exceeded")

U=transmittance(dx0, N, m, l)
Uf=Fresnel(U, w_l, dx0, dx, z)


I=(np.abs(Uf)**2)                            #Intensity
angle=np.angle(Uf)                           #Phase

x=N*dx
y=N*dx

plt.figure(1)
plt.imshow(U)
plt.imsave("Diffraction_grating.png",U)

plt.figure(2)
plt.imshow(I)
plt.imsave("Talbot_effectInt.png",I, cmap='gray')

plt.figure(3)
plt.imshow(angle)
plt.imsave("Talbot_effectPhase.png",angle, cmap='gray')

toc=time.time()
print("time: ",toc-tic," sec")

