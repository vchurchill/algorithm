# import necessary packages
import numpy as np
#import matplotlib.pyplot as plt
from functions import *

''' tree '''
# of levels (counting level 0)
L = 4

# number of intervals per level
I = np.power(2,L-1)

# number of sources
N = 100

a = np.zeros(shape=(I+1,1))
b = np.zeros(shape=(I+1,1))

# root box
a1 = 0
a2 = 2
b1 = -1
b2 = 1

# sources
sources = np.random.rand(N,2)
for i in range(0,N):
  sources[i,0] = sources[i,0]*(b1-a1)+a1
  sources[i,1] = sources[i,1]*(b2-a2)+a2

# assign charge to each source, +1 or -1
sigma = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma[i] = (np.random.randint(0,2)*2)-1

rint = np.zeros(shape=(L,L))
zint = np.zeros(shape=(L,L))
# current level
for i in range(0,L):
  for j in range(0,L):
    rint[i,j] = (a1,a2)
    zint[i,j] = (b1,b2)