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
    Nzones=5       #Number of Fresnel zones
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
z=900                   #0.9 mm
N=M=int(256/2)     #Number of pixels



lim=N*(deltax0**2)/lamda  #Limit of z in Angular Spectrum
print ("lim:",lim)
if z>lim:
    print("z limit exceeded")



image=cv2.imread("totoro.png",0)


#Despectroangular(image,1000*lamda,lamda,deltaxprim,deltayprim)
U=Umatrix(z, lamda, deltax0, N)

tic=time.time()
Uz=Despectroangular(U,z,lamda,deltax0,deltay0)
toc=time.time()

Tfft=toc-tic

tic=time.time()
UzDFT=DespectroangularDFT(U,z,lamda,deltax0)
toc=time.time()

Tdft=toc-tic
#FFT
I=(np.abs(Uz)**2)                            #Intensity
angle=np.angle(Uz)                           #Phase

#DFT
Idft=(np.abs(UzDFT)**2)                            #Intensity
angledft=np.angle(UzDFT)                           #Phase

x=N*deltax
y=N*deltay

plt.figure(1)
plt.imshow(I, extent=[-x,x,-y,y])

plt.figure(2)
plt.imshow(I)
plt.imsave("AS_FFT_Int.png",I, cmap='gray')

plt.figure(3)
plt.imshow(angle)
plt.imsave("AS_FFT_Phase.png",angle, cmap='gray')

plt.figure(4)
plt.imshow(Idft)
plt.imsave("AS_FFT_IntDFT.png",Idft, cmap='gray')

plt.figure(5)
plt.imshow(angledft)
plt.imsave("AS_FFT_PhaseDFT.png",angledft, cmap='gray')



print("time: ",Tdft/Tfft," sec")
print("Pixels: ",2*N)

# plt.figure()
# plt.imshow(np.abs(I1-deltaxprim*I2),cmap="gray")
# plt.show()


