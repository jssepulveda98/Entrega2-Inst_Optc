import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

#Functions

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

def Despectroangular(U,z,lamda,dx0,dy0):
    
    Uz=np.fft.fftshift(np.fft.fftn(U))
    
    
    N,M=np.shape(U)
    
    x=np.arange(-int(M/2),int(M/2),1)
    y=np.arange(-int(N/2),int(N/2),1)
    X,Y=np.meshgrid(x,y)
    
    fx=X*(1/(M*dx0))
    fy=Y*(1/(N*dy0))
    
    k=(2*np.pi)/lamda
    
    Prop=np.exp(1j*z*(k)*((1 -((lamda**2)*(fx**2 +fy**2)))**0.5)) 
    
    Uz=Uz*Prop
    
        
    Uz=np.fft.ifftn(Uz)
    
        
    return Uz
    
    
"""
U=incident wave
z=entrance plane to detector plane distance
lamda= wavelength
deltax=deltay=pixel size detector plane
deltax0=deltay0=pixel size entrance plane
M=number of pixels in the x axis
N=number of pixels in the y axis
(M=N)
MxN=number of pixels
"""
        
lamda=0.633          #(633nm orange/red)   #All units in um
deltax0=2.5              #2.5um
deltay0=2.5              #2.5um
deltax=deltax0
deltay=deltay0
#z=2500                   #2.5 mm
N=M=2048                 #Number of pixels
#Diffraction grating parameters
m=1               #Contrast factor
l=80              #Period
z=3*(l**2)/lamda
print('z:',z)

tic=time.time()

lim=N*(deltax0**2)/lamda  #Limit of z in Angular Spectrum
print ("lim:",lim)
if z>lim:
    print("z limit exceeded")


U=transmittance(deltax0, N, m, l)
Uz=Despectroangular(U,z,lamda,deltax0,deltay0)

I0=(np.abs(U)**2)                            #Intensity

I=(np.abs(Uz)**2)                            #Intensity
angle=np.angle(Uz)                           #Phase


x=N*deltax
y=N*deltay

plt.figure(1)
plt.imshow(I0)
plt.imsave("Diffraction_grating(AS).png",I0, cmap='gray')

plt.figure(2)
plt.imshow(I)
plt.imsave("Talbot_effectInt(AS).png",I, cmap='gray')


toc=time.time()
print("time: ",toc-tic," sec")