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
n=10

''' specify source interval '''
# parent interval
Pa = 2
Pb = 4

# Target interval
a = 3
b = 4

# target point in that interval
point = 3.2

# parent interaction list interval
Pc = 6
Pd = 8

# child interaction list intervals
C1a = 0
C1b = 1

C2a = 1
C2b = 2

C3a = 5
C3b = 6

''' populate boxes '''

# create sources
sourcesP = np.random.rand(2*N,1)*(Pd-Pc) + Pc
sourcesC1 = np.random.rand(N,1)*(C1b-C1a) + C1a
sourcesC2 = np.random.rand(N,1)*(C2b-C2a) + C2a
sourcesC3 = np.random.rand(N,1)*(C3b-C3a) + C3a

'''
sourcesy = np.zeros(shape=(N,1))
plt.scatter(sources,sourcesy,color='black')
plt.grid()
plt.show()
'''
# create some charge (density) for each source
# they can be +1 or -1
# notice how we handle each child box's source charges separately
sigmaP = np.zeros(shape=(2*N,1))
for i in range(0,2*N):
  sigmaP[i] = (np.random.randint(0,2)*2)-1

sigmaC1 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC1[i] = (np.random.randint(0,2)*2)-1

sigmaC2 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC2[i] = (np.random.randint(0,2)*2)-1

sigmaC3 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC3[i] = (np.random.randint(0,2)*2)-1

''' these are the weights for each interaction list interaction '''

# each Chebyshev node, m, gets a weight
def WP(m):
  sum = 0
  for j in range(0,2*N):
    sum += S(n,nodes(n,Pc,Pd)[m],sourcesP[j],Pc,Pd)*sigmaP[j]
  return sum

def WC1(m):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,C1a,C1b)[m],sourcesC1[j],C1a,C1b)*sigmaC1[j]
  return sum

def WC2(m):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,C2a,C2b)[m],sourcesC2[j],C2a,C2b)*sigmaC2[j]
  return sum

def WC3(m):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,C3a,C3b)[m],sourcesC3[j],C3a,C3b)*sigmaC3[j]
  return sum

''' compute multipole to local estimate '''
def gP(l):
  sum = 0
  for m in range(0,n):
    sum += log(nodes(n,Pa,Pb)[l],nodes(n,Pc,Pd)[m])*WP(m)
  return sum

def gC(l):
  sum = 0
  for m in range(0,n):
    sum += log(nodes(n,a,b)[l],nodes(n,C1a,C1b)[m])*WC1(m) + log(nodes(n,a,b)[l],nodes(n,C2a,C2b)[m])*WC2(m) + log(nodes(n,a,b)[l],nodes(n,C3a,C3b)[m])*WC3(m)
  return sum

''' compute estimate for local expansion '''
''' note there is no local to local contribution b/c parent has no interaction list '''
def f(l):
  sum = gC(l)
  for lprime in range(0,n):
    sum += gP(lprime)*S(n,nodes(n,a,b)[l],nodes(n,Pa,Pb)[lprime],Pa,Pb)
  return sum

fest = 0
for l in range(0,n):
  fest += f(l)*S(n,nodes(n,a,b)[l],point,a,b)

print("Estimated potential:")
print(fest)

''' computation of actual local expansion '''
fact = 0
for j in range(0,2*N):
  fact += log(point,sourcesP[j])*sigmaP[j]
for j in range(0,N):
  fact += log(point,sourcesC1[j])*sigmaC1[j] + log(point,sourcesC2[j])*sigmaC2[j] + log(point,sourcesC3[j])*sigmaC3[j]

print("Actual potential:")
print(fact)