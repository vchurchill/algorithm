# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt
from functions import *

''' specify source number and node number '''

# N is the number of sources in each interval
N = 5

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=8

''' specify source interval '''
# source parent
a = 0
b = 2

# source children
C1a = a
C1b = a + (b-a)/4
C2a = a + (b-a)/4
C2b = (a+b)/2
C3a = (a+b)/2
C3b = a + 3*(b-a)/4
C4a = a + 3*(b-a)/4
C4b = b

# target
c = 3
d = 5

''' populate boxes '''

# create sources in child boxes separately
sources1 = np.random.rand(N,1)*(C1b-C1a) + C1a
sources2 = np.random.rand(N,1)*(C2b-C2a) + C2a
sources3 = np.random.rand(N,1)*(C3b-C3a) + C3a
sources4 = np.random.rand(N,1)*(C4b-C4a) + C4a

# create target
point = 4.5

'''
sourcesy = np.zeros(shape=(N,1))
plt.scatter(sources1,sourcesy,color='black')
plt.scatter(sources2,sourcesy,color='red')
plt.scatter(sources3,sourcesy,color='blue')
plt.scatter(sources4,sourcesy,color='yellow')
plt.scatter(point,0,color='green')
plt.grid()
plt.show()
'''

# create some charge (density) for each source
# they can be +1 or -1
# notice how we handle each child box's source charges separately
sigma1 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma1[i] = (np.random.randint(0,2)*2)-1

sigma2 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma2[i] = (np.random.randint(0,2)*2)-1

sigma3 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma3[i] = (np.random.randint(0,2)*2)-1

sigma4 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma4[i] = (np.random.randint(0,2)*2)-1

''' these are the weights for each child box '''

# each Chebyshev node, m, gets a weight
def W1(m,a,b):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,a,b)[m],sources1[j],a,b)*sigma1[j]
  return sum

def W2(m,a,b):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,a,b)[m],sources2[j],a,b)*sigma2[j]
  return sum
  
def W3(m,a,b):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,a,b)[m],sources3[j],a,b)*sigma3[j]
  return sum

def W4(m,a,b):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,a,b)[m],sources4[j],a,b)*sigma4[j]
  return sum

# PARENT
def W(m):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,a,b)[m],sources1[j],a,b)*sigma1[j] + S(n,nodes(n,a,b)[m],sources2[j],a,b)*sigma2[j] + S(n,nodes(n,a,b)[m],sources3[j],a,b)*sigma3[j] + S(n,nodes(n,a,b)[m],sources4[j],a,b)*sigma4[j]
  return sum

''' this is the M2M operation taking the W weights of the child boxes and translating to the parent '''
def West(m):
  sum = 0
  for mprime in range(0,n):
    sum += S(n,nodes(n,a,b)[m],nodes(n,C1a,C1b)[mprime],a,b)*W1(mprime,C1a,C1b) + S(n,nodes(n,a,b)[m],nodes(n,C2a,C2b)[mprime],a,b)*W2(mprime,C2a,C2b) + S(n,nodes(n,a,b)[m],nodes(n,C3a,C3b)[mprime],a,b)*W3(mprime,C3a,C3b) + S(n,nodes(n,a,b)[m],nodes(n,C4a,C4b)[mprime],a,b)*W4(mprime,C4a,C4b)
  return sum

print("weight from parent box calculated by itself")
wparent = 0
for m in range(0,n):
  wparent += W(m)
print(wparent)

for m in range(0,n):
  print(W(m))

print("weight from parent box calculated from child boxes and M2M")
wmtm = 0
for m in range(0,n):
  wmtm += West(m)
print(wmtm)

for m in range(0,n):
  print(West(m))
  
'''

estimate sum to N of f(x)=K(x,y(j))*sigma[j] by sum to n of K(x,y(m))*West(m,a,b)
def Kest1(l):
  Kest1 = 0
  for m in range(0,n):
    Kest1 += log(nodes(n,c,d)[l],nodes(n,a,b)[m])*West(m)
  return Kest1

def Kest2(l):
  Kest2 = 0
  for m in range(0,n):
    Kest2 += log(nodes(n,c,d)[l],nodes(n,a,b)[m])*W(m)
  return Kest2

print("kernel estimate first")
for l in range(0,n):
  print(Kest1(l))
  print(Kest2(l))

fest1 = 0
for l in range(0,n):
  fest1 += S(n,nodes(n,c,d)[l],point,c,d)*Kest1(l)

print("Estimated potential 1:")
print(fest1)

fest2 = 0
for l in range(0,n):
  fest2 += S(n,nodes(n,c,d)[l],point,c,d)*Kest1(l)

print("Estimated potential 2:")
print(fest2)



 estimate sum to N of f(x)=K(x,y(j))*sigma[j] by sum to n of K(x,y(m))*West(m,a,b)
fest3 = 0
for m in range(0,n):
  fest2 += log(point,nodes(n,a,b)[m])*West(m)

print("Estimated potential 3:")
print(fest3)

# compute actual sum to N of f(x)=K(x,y(j))*sigma[j]
fact = 0
for j in range(0,N):
  fact += log(point,sources1[j])*sigma1[j] + log(point,sources2[j])*sigma2[j] + log(point,sources3[j])*sigma3[j] + log(point,sources4[j])*sigma4[j]

print("Actual potential:")
print(fact)


print("weight estimate first")
for m in range(0,n):
  print(West(m))
  print(W(m))

Westimate = 0
for m in range(0,n):
  Westimate += West(m)
print("total weight estimate 1")
print(Westimate)

Wactual = 0
for m in range(0,n):
  Wactual += W(m)
print("total weight estimate 2")
print(Wactual)

'''