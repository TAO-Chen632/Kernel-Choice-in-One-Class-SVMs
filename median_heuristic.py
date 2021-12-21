import numpy as np
from sklearn.datasets import load_boston

# Get data
X1 = load_boston()['data'][:, [8, 10]]  # two clusters
X2 = load_boston()['data'][:, [5, 12]]  # "banana"-shaped

#Median heuristic of data set X1
n=len(X1)
list1=[]
for i in range(n):
    for j in range(i):
        list1.append(pow(((X1[i][0]-X1[j][0])**2+(X1[i][1]-X1[j][1])**2),0.5))
MH1=np.median(list1)
gamma1=1/(MH1**2)
print(gamma1)

#Median heuristic of data set X2
m=len(X2)
list2=[]
for i in range(m):
    for j in range(i):
        list2.append(pow(((X2[i][0]-X2[j][0])**2+(X2[i][1]-X2[j][1])**2),0.5))
MH2=np.median(list2)
gamma2=1/(MH2**2)
print(gamma2)




