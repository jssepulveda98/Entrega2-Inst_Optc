import numpy as np
import matplotlib.pyplot as plt
import cv2

#Functions


def DespectroangularF(U,z,lamda,delta):
    
    A=np.zeros(np.shape(U),dtype=np.complex64)
    Mue=np.shape(A)[0]
    p,q=np.shape(A)
    
    n=np.arange(-int(q/2),int(q/2),1)
    m=np.arange(-int(p/2),int(p/2),1)
    
    
    N,M=np.meshgrid(n,m)
    
    delta_f=1/(Mue*delta)
    
    k=2*np.pi/lamda
    
    for i in np.arange(len(N)):
        for j in np.arange(len(M)):
            A[i,j]=(delta**2)*np.sum(U*np.exp(-1j*(2*np.pi/Mue)*(i*N+j*M)))
            
            
    A=np.fft.fftshift(A)
    
    Az=A*np.exp(1j*z*k*((1 - ((lamda*delta_f)**2) *(N**2  +M**2))**0.5))
    
    
    Uz=np.zeros(np.shape(U),dtype=np.complex64)
    for i in np.arange(len(N)):
        for j in np.arange(len(M)):
            Uz[i,j]=(delta_f**2)*np.sum(Az*np.exp(1j*(2*np.pi/Mue)*(i*N+j*M)))
            
    Uz=np.fft.fftshift(Uz)
    
    
    return Uz
    
    
def Despectroangular(U,z,lamda,dx_f,dy_f):
    
    Uz=np.fft.fftshift(np.fft.fftn(U))
    
    
    N,M=np.shape(U)
    
    x=np.arange(-int(M/2),int(M/2),1)
    y=np.arange(-int(N/2),int(N/2),1)
    X,Y=np.meshgrid(x,y)
    
    fx=X*(1/(M*dx_f))
    fy=Y*(1/(N*dy_f))
    
    k=2*np.pi/lamda
    
    Prop=np.exp(1j*z*(k)*((1 -(fx**2 +fy**2))**0.5)) 
    
    Uz=Uz*Prop
    
        
    Uz=np.fft.ifftn(Uz)
    
        
    return Uz
    
    




image=cv2.imread("totoro.png",0)


#All units in micrometers

lamda=0.633
deltaxprim=2.5 
deltayprim=2.5






#Despectroangular(image,1000*lamda,lamda,deltaxprim,deltayprim)

I1=np.abs(Despectroangular(image,100*lamda,lamda,deltaxprim,deltayprim))**2
I2=np.abs(DespectroangularF(image,250*lamda,lamda,deltaxprim))**2


plt.figure()
plt.imshow(I1,cmap="gray")
plt.figure()
plt.imshow(I2,cmap="gray")
plt.figure()
plt.imshow(np.abs(I1-I2),cmap="gray")
plt.show()


