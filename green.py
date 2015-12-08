import numpy as np
import math
import cmath
import scipy
from scipy import special

def X(r1,z1,r2,z2):
  return (np.square(r1)+np.square(r2)+np.square(z1-z2))/(2*r1*r2)

def q(r1,z1,r2,z2,n):
  return ((np.sqrt(math.pi)*scipy.special.gamma(n+1/2))/(cmath.exp((n+1/2)*cmath.log(2))*scipy.special.gamma(n+1)))*(1/(cmath.exp((n+1/2)*cmath.log(X(r1,z1,r2,z2)))))*(scipy.special.hyp2f1(((n+1/2)/2),((n+3/2)/2),(n+1),1/(cmath.exp(2*cmath.log(X(r1,z1,r2,z2))))))

def Laplace2D(r1,z1,r2,z2):
  return (1/(2*math.pi))*np.log(math.sqrt(np.square(r1-r2)+np.square(z1-z2)))

def Laplace3D(r1,z1,r2,z2):
  return 1/(4*math.pi*np.power((np.square(r1-r2)+np.square(z1-z2)),1/2))
  
def LaplaceF2D(r1,z1,r2,z2,n):
  return q(r1,z1,r2,z2,n)*cmath.exp((-1/2)*cmath.log(8*math.pi*r1*r2))