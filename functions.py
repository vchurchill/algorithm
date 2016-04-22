import numpy as np
from numpy import matlib
from numpy import linalg
import math
import cmath
import scipy
from scipy import special
'''
def X(r1,z1,r2,z2):
  return (np.square(r1)+np.square(r2)+np.square(z1-z2))/(2*r1*r2)

def q(r1,z1,r2,z2,n):
  return np.sqrt(math.pi)*np.power(2,(-n-1/2))*scipy.special.gamma(n+1/2)*np.power(X(r1,z1,r2,z2),(-n-1/2))*scipy.special.hyp2f1(n/2+3/4,n/2+3/4,n+1,np.power(X(r1,z1,r2,z2),-2))/scipy.special.gamma(n+1)

def LaplaceMode(r1,z1,r2,z2,n):
  return q(r1,z1,r2,z2,n)*np.power(r2/r1,1/2)/np.sqrt(2*math.pi)
'''
# define tikhonov regularization function
def tikh(M,p):
# regularization parameter
  alpha = np.power(10,-12)
#  identity matrix
  I = np.matlib.identity(p)
  return np.dot(np.linalg.inv(alpha*I+np.dot(np.matrix.transpose(M),M)),np.matrix.transpose(M))


def Laplace2D(r1,z1,r2,z2):
  return (1/(2*math.pi))*np.log(math.sqrt(np.square(r1-r2)+np.square(z1-z2)))

#def T(n,x):
#  T(0,x)=1
#  T(1,x)=x
#  return 2*x*T(n-1,x)-T(n-2,x)

#def Laplace3D(r1,t1,z1,r2,t2,z2):
#  return 1/(4*math.pi*np.power((np.square(r1-r2)+np.square(z1-z2)),1/2))
