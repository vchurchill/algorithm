# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt

''' define Chebyshev functions and log kernel '''

# Chebyshev polynomial function
def T(n,x,a,b):
  return np.cos(n*np.arccos(((2/(b-a))*(x-((a+b)/2)))))

# compute Chebyshev nodes in interval -1,1
def nodes(n,a,b):
  nodes = np.zeros(shape=(n,1))
  for i in range(0,n):
    nodes[i] = (a+b)/2+ ((b-a)/2)*np.cos(((2*(i+1)-1)*np.pi)/(2*n))
  return nodes

# define other Chebyshev polynomial function
def S(n,x,y,a,b):
  k=1
  sum=0
  while k <= (n-1):
    sum+=T(k,x,a,b)*T(k,y,a,b)
    k+=1
  return 1/n + (2/n)*sum

# define log kernel
def log(r,z):
  return np.log(np.fabs(r-z))

''' specify source number and node number '''

# N is the number of sources in each interval
N = 40

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=8

''' specify source interval '''
# interval that we wanna get a local expansion for
a = 1
b = 2

# interval that we have a multipole expansion for
c = 3
d = 4

''' populate boxes '''

# create sources in child boxes separately
sourcesm = np.random.rand(N,1)*(d-c) + c
sourcesl = np.random.rand(N,1)*(b-a) + a
'''
sourcesy = np.zeros(shape=(N,1))
plt.scatter(sources,sourcesy,color='black')
plt.grid()
plt.show()
'''
# create some charge (density) for each source
# they can be +1 or -1
# notice how we handle each child box's source charges separately
sigmam = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmam[i] = (np.random.randint(0,2)*2)-1
sigmal = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmal[i] = (np.random.randint(0,2)*2)-1

''' these are the weights for each interaction list interaction '''

# each Chebyshev node, m, gets a weight
def Wm(m):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,c,d)[m],sourcesm[j],c,d)*sigmam[j]
  return sum

def Wl(m):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,a,b)[m],sourcesl[j],a,b)*sigmal[j]
  return sum

''' compute local expansion estimate '''
def g(l):
  g1 = 0
  for m in range(0,n):
    g1 += log(nodes(n,a,b)[l],nodes(n,c,d)[m])*W(m)
  return g1

def gest(l):
  g2 = 0
  for m in range(0,n):
    g2 += 

''' print all the nodes '''
for l in range(0,n):
  print(g(l))

''' other computation of local expansion '''

