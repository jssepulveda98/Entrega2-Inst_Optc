import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

#Functions

def Umatrix(z, lamda, deltax0, N):
    """
    Incident wave and transmittance function
    In this case: plane wave and circular aperture 
    """
    deltay0=deltax0
    x=np.arange(-N/2,N/2)
    y=np.arange(-N/2,N/2)
    x,y=np.meshgrid(x,y)
    Nzones=10       #Number of Fresnel zones
    lim=Nzones*lamda*z
    U_matrix=(deltax0*x)**2 + (deltay0*y)**2
    U_matrix[np.where(U_matrix<=lim)]=1
    U_matrix[np.where(U_matrix>lim)]=0

    return U_matrix

def DespectroangularDFT(U,z,lamda,delta):
    """
    Returns
    -------
    Uz : Diffracted image at Z prop distance

    """
    
    A=np.zeros(np.shape(U),dtype=np.complex64)
    Mue=np.shape(A)[0]
    p,q=np.shape(A)
    
    n=np.arange(-int(q/2),int(q/2),1)
    m=np.arange(-int(p/2),int(p/2),1)
    
    
    N,M=np.meshgrid(n,m)
    
    delta_f=1/(Mue*delta)
    
    k=2*np.pi/lamda
    
    #DFT
    
    for i in np.arange(len(N)): 
        for j in np.arange(len(M)):
            A[i,j]=(delta**2)*np.sum(U*np.exp(-1j*(2*np.pi/Mue)*(i*N+j*M)))
            
            
    A=np.fft.fftshift(A)
    
    #Prop
    Az=A*np.exp(1j*z*k*((1 - ((lamda*delta_f)**2) *(N**2  +M**2))**0.5))
    
    #IDFT
    
    Uz=np.zeros(np.shape(U),dtype=np.complex64)
    for i in np.arange(len(N)): 
        for j in np.arange(len(M)):
            Uz[i,j]=(delta_f**2)*np.sum(Az*np.exp(1j*(2*np.pi/Mue)*(i*N+j*M)))
            
    Uz=np.fft.fftshift(Uz)
    
    
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
z=2500                   #2.5 mm
N=M=128                 #Number of pixels

tic=time.time()

lim=N*(deltax0**2)/lamda  #Limit of z in Angular Spectrum
print ("lim:",lim)
if z>lim:
    print("z limit exceeded")


U=Umatrix(z, lamda, deltax0, N)
Uz=DespectroangularDFT(U,z,lamda,deltax0)
toc=time.time()
#DFT
I=(np.abs(Uz)**2)                            #Intensity
angle=np.angle(Uz)                           #Phase


x=N*deltax
y=N*deltay

plt.figure(1)
plt.imshow(I, extent=[-x,x,-y,y])

plt.figure(2)
plt.imshow(I)
plt.imsave("AS_DFT_Int_Fresnelzones.png",I, cmap='gray')

plt.figure(3)
plt.imshow(angle)
plt.imsave("AS_DFT_Phase_Fresnelzones.png",angle, cmap='gray')


print("time: ",toc-tic," sec")