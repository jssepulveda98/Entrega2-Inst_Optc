import numpy as np
import matplotlib.pyplot as plt
import cv2

#Functions

def Sum(U,N,p,q,phase):
    
    x,y=np.shape(U)[0],np.shape(U)[1]
    sumx=0
    
    for m in np.arange(x):
        for n in np.arange(y):
            sumx+=U[n,m]*np.exp(phase*(2*np.pi/N)*(p*n +q*m))
            
    
    return sumx


def DespectroangularF(delta,delta_f,U,N,lamda,z):
    
    A=np.zeros(np.shape(U))
    P,Q=np.shape(A)[0],np.shape(A)[1]
    
    for p in np.arange(P):
        for q in np.arange(Q):
            
            A[p,q]=(delta**2)*Sum(U,N,p,q,-1j)
            
    A_z=np.zeros(np.shape(U))
    
    for p in np.arange(P):
        for q in np.arange(Q):
            
            A_z[p,q]=A[p,q]*np.exp(1j*z*(2*np.pi/lamda)*((1- ((lamda*delta_f)**2)*(p**2 +q**2))**0.5))
            
    U_z=np.zeros(np.shape(U))
    
    for p in np.arange(P):
        for q in np.arange(Q):
            
            U_z[p,q]=(delta_f**2)*Sum(A_z,N,p,q,1j)
            
    
    return U_z
    
    
def Despectroangular(U,z,lamda,deltau,deltav):
    Uz=(1/(1j*lamda))*np.fft.fft2(U*deltau*deltav)
    Uz=np.fft.fftshift(Uz)
    
    Uz=Uz*np.exp(1j*z*(2*np.pi/lamda)*((1- (deltau**2 +deltav**2))**0.5))
    
    Uz=np.fft.ifft2(Uz)
    
    return Uz
    
    




image=cv2.imread("cameraman.png",0)

N=256
lamda=0.633
deltaxprim=2.5 
deltayprim=2.5



#Constrains of pixel sizes do to the Fourier transform discretization
deltau=(lamda)/(2*N*deltaxprim) 
deltav=(lamda)/(2*N*deltayprim)


plt.imshow(np.abs(Despectroangular(image,10000*lamda,lamda,deltau,deltav))**2)
plt.show()


