import numpy as np
import scipy
from scipy import special

def q(n,r1,z1,r2,z2):
  ((np.sqrt(math.pi)*scipy.special.gamma(n+1/2))/(np.power(2,n+1/2)*scipy.special.gamma(n+1)))*np.power(((np.square(r1)+np.square(r2)+np.square(z1-z2))/(2*r1*r2)),-n-1/2)
  *scipy.special.hyp2f1((n+1/2)/2,(n+3/2)/2,n+1,np.power(((np.square(r1)+np.square(r2)+np.square(z1-z2))/(2*r1*r2)),-2))