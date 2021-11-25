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

def Umatrix(z, w_l, dx0, N):
    """
    Incident wave and transmittance function
    In this case: plane wave and circular aperture 
    """
    dy0=dx0
    x=np.arange(-N/2,N/2)
    y=np.arange(-N/2,N/2)
    x,y=np.meshgrid(x,y)
    Nzones=20       #Number of Fresnel zones
    lim=Nzones*w_l*z
    U_matrix=(dx0*x)**2 + (dy0*y)**2
    U_matrix[np.where(U_matrix<=lim)]=1
    U_matrix[np.where(U_matrix>lim)]=0

    return U_matrix

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
w_l= wavelentgth
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
z=21000             #2.5 mm

tic=time.time()

"-----PADDING-----" #If needed 
"""
width=height=512
r=int(512/2)
U_0 = cv2.copyMakeBorder(U_0,r,r,r,r,cv2.BORDER_CONSTANT)"""
""" FINALIZA PADDING """


dx=w_l*z/(dx0*N)
print ("dx:",dx)
lim=N*(dx0**2)/w_l  #Limit of z in FT
print ("lim:",lim)
if z<lim:
    print("z limit exceeded")

U=Umatrix(z, w_l, dx0, N)
Uf=Fresnel(U, w_l, dx0, dx, z)


I=(np.abs(Uf)**2)                            #Intensity
angle=np.angle(Uf)                           #Phase

x=N*dx
y=N*dx

plt.figure(1)
plt.imshow(I, extent=[-x,x,-y,y])

plt.figure(2)
plt.imshow(I)
plt.imsave("FresnelInt2.png",I, cmap='gray')

plt.figure(2)
plt.imshow(angle)
plt.imsave("FresnelPhase2.png",angle, cmap='gray')

toc=time.time()
print("time: ",toc-tic," sec")

