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
N=256
lamda=0.633
deltaxprim=2.5 
deltayprim=2.5








plt.imshow(np.abs(Despectroangular(image,1000*lamda,lamda,deltaxprim,deltayprim))**2)
plt.show()


