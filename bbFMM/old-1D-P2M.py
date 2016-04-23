# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt

''' define Chebyshev functions and log kernel '''

# Chebyshev polynomial function
def T(n,x):
  return np.cos(n*np.arccos(x))

# compute Chebyshev nodes in interval -1,1
def nodes(n):
  nodes = np.zeros(shape=(n,1))
  for i in range(0,n):
    nodes[i] = np.cos(((2*(i+1)-1)*np.pi)/(2*n))
  return nodes

# define other Chebyshev polynomial function
def S(n,x,y):
  k=1
  sum=0
  while k <= (n-1):
    sum+=T(k,x)*T(k,y)
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

a = -1
b = 1

''' populate boxes '''

# create sources in box
sources = np.random.rand(N,1)*(b-a) + a

'''
sourcesy = np.zeros(shape=(N,1))
plt.scatter(sources,sourcesy,color='black')
plt.grid()
plt.show()
'''
# create target
point = 3.6

# create some charge (density) for each source
# they can be +1 or -1
sigma = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma[i] = (np.random.randint(0,2)*2)-1

# each Chebyshev node, m, gets a weight
def W(m):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n)[m],sources[j])*sigma[j]
  return sum

# estimate sum to N of f(x)=K(x,y(j))*sigma[j] by sum to n of K(x,y(m))*W(m)
fest = 0
for m in range(0,n):
  fest += log(point,nodes(n)[m])*W(m)

print("Estimated potential:")
print(fest)

# compute actual sum to N of f(x)=K(x,y(j))*sigma[j]
fact = 0
for j in range(0,N):
  fact += log(point,sources[j])*sigma[j]

print("Actual potential:")
print(fact)