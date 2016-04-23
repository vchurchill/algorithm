# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt

''' define Chebyshev functions and log kernel '''

# Chebyshev polynomial function
def T(n,x,a,b):
  return np.cos(n*np.arccos(((2/(b-a))*(x-((a+b)/2)))))
''' note domain of arccos in [-1,1] so that's why this translation is happening '''
# compute Chebyshev nodes in interval -1,1
def nodes(n,a,b):
  nodes = np.zeros(shape=(n,1))
  for i in range(0,n):
    nodes[i] = (a+b)/2+ ((b-a)/2)*np.cos(((2*(i+1)-1)*np.pi)/(2*n))
  return nodes

# define other Chebyshev polynomial function
def S(n,x,y,a1,b1,a2,b2):
  k=1
  sum=0
  while k <= (n-1):
    sum+=T(k,x,a1,b1)*T(k,y,a2,b2)
    k+=1
  return 1/n + (2/n)*sum

# define multivariable version of above S function
def R(n,r1,z1,r2,z2,a1,b1,a2,b2):
  return S(n,r1,r2,a1,b1,a2,b2)*S(n,z1,z2,a1,b1,a2,b2)

# define log kernel
def log(r1,z1,r2,z2):
  return np.log(np.sqrt(np.square(r1-r2)+np.square(z1-z2)))


''' specify source number and node number '''

# N is the number of sources in each interval
N = 40

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=8

''' specify source interval '''

a1 = -1
b1 = 1
a2 = -1
b2 = 1
c1 = 3
d1 = 5
c2 = 3
d2 = 5

''' populate boxes '''

# create sources in box
sources = np.random.rand(N,2)
for i in range(0,N):
  sources[i,0] = sources[i,0]*(b1-a1)+a1
  sources[i,1] = sources[i,1]*(b2-a2)+a2

# create target
point = [3.1,3.3]
'''
for i in range(0,N):
  plt.scatter(sources[i,0],sources[i,1],color='black')
  plt.scatter(point[0],point[1],color='green')
plt.grid()
plt.show()
'''
# create some charge (density) for each source
# they can be +1 or -1
sigma = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma[i] = (np.random.randint(0,2)*2)-1

# each Chebyshev node, m, gets a weight
def W(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],sources[j,0],sources[j,1],a1,b1,a2,b2)*sigma[j]
  return sum

# estimate sum to N of f(x)=K(x,y(j))*sigma[j] by sum to n of K(x,y(m))*W(m)
def Kest(l1,l2):
  Kest = 0
  for m1 in range(0,n):
    for m2 in range(0,n):
      Kest += log(nodes(n,c1,d1)[l1],nodes(n,c2,d2)[l2],nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2])*W(m1,m2)
  return Kest

fest1 = 0
for l1 in range(0,n):
  for l2 in range(0,n):
    fest1 += R(n,nodes(n,c1,d1)[l1],nodes(n,c2,d2)[l2],point[0],point[1],c1,d1,c2,d2)*Kest(l1,l2)

print("Estimated potential 1:")
print(fest1)

# estimate sum to N of f(x)=K(x,y(j))*sigma[j] by sum to n of K(x,y(m))*W(m)
fest2 = 0
for m1 in range(0,n):
  for m2 in range(0,n):
    fest2 += log(point[0],point[1],nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2])*W(m1,m2)

print("Estimated potential 2:")
print(fest2)

# compute actual sum to N of f(x)=K(x,y(j))*sigma[j]
fact = 0
for j in range(0,N):
  fact += log(point[0],point[1],sources[j,0],sources[j,1])*sigma[j]

print("Actual potential:")
print(fact)
