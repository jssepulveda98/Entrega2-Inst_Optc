"""
Fresnel Transform using TTF

ESQUEMA
1. Calculate U'[n,m,0] multiplying by a phase
2. Calculate U''[n,m,z] using FFT
3. Calculate U[n,m,z] multiplying by a scaling function
4. Reordenar el campo
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2



def Fresnel(Uin, w_l, dx0, z):
    "-----Step 1------"
    k=2*np.pi/w_l
    N,M=np.shape(Uin)
    x=np.arange(-N/2,N/2,1)
    y=np.arange(-M/2,M/2,1)
    X,Y=np.meshgrid(x,y)
    phase=np.exp((1j*k)/(2*z)*(((X*dx0)**2) + ((Y*dx0)**2)))
    U1=Uin*phase
    "-----Step 2-----"
    dx=w_l*z/(dx0*N)
    print (dx)
    #X=X*(1/(M*dx0))
    #Y=Y*(1/(N*dx0))
    Uf=np.fft.fftshift(np.fft.fft2(U1*dx0**2))
    "-----Step 3-----"
    X=X*dx
    Y=Y*dx
    c1=np.exp(1j*k*z)/(1j*w_l*z)
    Uf=Uf*c1*np.exp(1j*(k/2*z)*((X)**2 + (Y)**2))
    
    return Uf

"-----Physical array-----"
        
w_l=633          #(633nm orange/red) #All units in um
dx0=2000        #2um
N=M=int(512/2)
#z=1*N*(dx0**2)/w_l  #Condition of z in FT
z=32*1e5   #3.2 mm
U_0=cv2.imread('cameraman.png',0)

"-----PADDING-----" #If needed 
"""
width=height=512
r=int(512/2)
U_0 = cv2.copyMakeBorder(U_0,r,r,r,r,cv2.BORDER_CONSTANT)"""
""" FINALIZA PADDING """

print (z)
Uf=Fresnel(U_0, w_l, dx0, z)
I1=np.log(np.abs((Uf)**2))

plt.figure(1)
plt.imshow(U_0, extent=[-N,N,-N,N])


plt.figure(2)
plt.imshow(I1)
plt.imsave("TF.png",I1, cmap='gray')


