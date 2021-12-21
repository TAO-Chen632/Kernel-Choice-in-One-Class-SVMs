import numpy as np
from numpy.linalg import *
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# Get data
X1 = load_boston()['data'][:, [8, 10]]  # two clusters
X2 = load_boston()['data'][:, [5, 12]]  # "banana"-shaped
# The parameters
D=2
n=len(X1)   #n=len(X1)=len(X2)=506
m=4
Z1=np.array([[4.,18.9],[4.,17.6],[5.,18.7],[5.,17.0]])
Z2=np.array([[6.03427,14.237],[6.14718,11.2608],[6.31653,8.82576],[6.42944,7.60823]])
tau=1
# The functions to be used
def k(x,y,theta):
    return math.exp((-1*norm((x-y),ord=2)**2)/(2*pow(theta,2)))

def R(x,y,theta):
    def r(x,y,theta):
        return math.pi*pow(theta,2)*math.exp((-1*norm(x-y)**2)/(4*theta**2))
    return np.array([[r(x[i],y[j],theta) for j in range(len(y))] for i in range(len(x))])

def miuhatvector(z,theta):
    def miuhatpoint(x,theta):
        sum=0
        for i in range(n):
            sum+=k(X1[i],x,theta)
        return sum/n
    return np.array([miuhatpoint(z[i],theta) for i in range(len(z))])

def K(x,y,theta):
    def phi(x,y,theta):
        return np.array([k(y,x[i],theta) for i in range(len(x))])
    return np.array([phi(x,y[j],theta) for j in range(len(y))])

def log_marginal_likelihood(x,z,theta):
    def log_likehood_normal(x,z,theta):
        return -0.5*(math.log(det(R(z,z,theta)+(pow(tau,2)/n)*np.identity(m)))+\
              np.dot(np.dot(miuhatvector(z,theta),\
                     inv(R(z,z,theta)+(pow(tau,2)/n)*np.identity(m))),\
                     np.transpose(miuhatvector(z,theta)))+\
              (1/pow(tau,2))*pow(norm(K(x,z,theta)),2)-\
              (n/pow(tau,2))*pow(norm(miuhatvector(z,theta)),2)+\
              m*math.log(n)+m*(n-1)*math.log(tau**2)+m*n*math.log(2*math.pi))
    def gamma(pointx,vectorz,theta):
        array=np.identity(D)
        for i in range(D):
            for j in range(D):
                list=[]
                for l in range(m):
                    list.append((pow(k(pointx,vectorz[l],theta),2)*\
                                 (pointx[i]-vectorz[l][i])*\
                                 (pointx[j]-vectorz[l][j]))/pow(theta,4))
                array[i][j]=sum(list)
        return pow(det(array),1/2)
    logsum=0
    for i in range(n):
        logsum+=math.log(gamma(x[i],z,theta))
    value=logsum+log_likehood_normal(x,z,theta)
    return value

#Plot figure for the first data set
tau=2
x=[]
y=[]
valuepoint=6
step=0.05
while valuepoint<=9:
    x.append(valuepoint)
    y.append(log_marginal_likelihood(X1,Z1,valuepoint))
    valuepoint+=step
plt.figure(1)
plt.plot(x,y,color='blue')
plt.title('The marginal pseudolikelihood function for the first data set')
plt.xlabel('the bandwidth parameter {}'.format(chr(952)))
plt.ylabel('the marginal pseudolikelihood')
plt.xticks(np.linspace(6,9,13))
#plt.savefig('BLKE-BH1.pdf',dpi=600,format='pdf')

#Plot figure for the second data set
tau=1/3
x=[]
y=[]
valuepoint=2.5
step=0.05
while valuepoint<=5:
    x.append(valuepoint)
    y.append(log_marginal_likelihood(X2,Z2,valuepoint))
    valuepoint+=step
plt.figure(2)
plt.plot(x,y,color='green')
plt.title('The marginal pseudolikelihood function for the second data set')
plt.xlabel('the bandwidth parameter {}'.format(chr(952)))
plt.ylabel('the marginal pseudolikelihood')
plt.xticks(np.linspace(2.5,5,11))
#plt.savefig('BLKE-BH2.pdf',dpi=600,format='pdf')
plt.show()
