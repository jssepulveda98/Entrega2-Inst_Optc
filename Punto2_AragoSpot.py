import matplotlib.pyplot as plt
import numpy as np
import cv2


#Funtions

def Umatrix(z, w_l, dx0, N):
    """
    Incident wave and transmittance function
    In this case: plane wave and circular obstacle 
    """
    dy0=dx0
    x=np.arange(-N/2,N/2)
    y=np.arange(-N/2,N/2)
    x,y=np.meshgrid(x,y)
    Nzones=7       #Number of Fresnel zones
    lim=Nzones*w_l*z
    U_matrix=(dx0*x)**2 + (dy0*y)**2
    U_matrix[np.where(U_matrix<=1*lim)]=0
    U_matrix[np.where(U_matrix>1*lim)]=1
    U_matrix2=(dx0*x)**2 + (dy0*y)**2
    U_matrix2[np.where(U_matrix2<=6*lim)]=0
    U_matrix2[np.where(U_matrix2>6*lim)]=1
    U_matrix=U_matrix-U_matrix2
    
    

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






w_l=0.633          #(633nm orange/red)   #All units in um
dx0=2.5            #2.5um

N=M=2048           #Number of pixels
zF=(N*(dx0**2))/w_l +1000           #2.7 mm
zA=(N*(dx0**2))/w_l -1000
dx=w_l*zF/(dx0*N)
x=N*2.5
y=N*2.5
#Zf=PropDistance(1,w_l,1*w_l*zF) 
#Za=PropDistance(5,w_l,1*w_l*zA) 

Ias=np.abs(Despectroangular(Umatrix(zA,w_l,dx0,N),zA,w_l,dx0,dx0))**2

#If=np.abs(Fresnel(Umatrix(zF,w_l,dx0,N),w_l,dx0,dx,zF))**2


plt.figure()
plt.imshow(Umatrix(zA,w_l,dx0,N),cmap="gray", extent=[-x,x,-y,y])
plt.imsave("Aperture.png",Umatrix(zA,w_l,dx0,N), cmap='gray')

plt.figure()
plt.imshow(Ias,cmap="gray", extent=[-x,x,-y,y])
plt.imsave("AragoSpot7.png",Ias, cmap='gray')


#plt.figure()
#plt.imshow(If,cmap="gray", extent=[-x,x,-y,y])
plt.show()