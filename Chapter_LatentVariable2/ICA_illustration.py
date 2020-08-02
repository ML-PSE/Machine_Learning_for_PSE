##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          Illustration example for ICA vs PCA
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

#%% generate independent data
s1 = 2*np.sin(2*np.pi*8*np.arange(500)/500)
s2 = np.random.uniform(-2, 2, 500)

plt.figure()
plt.plot(s1)
plt.xlabel('sample #')
plt.ylabel('s1')

plt.figure()
plt.plot(s2)
plt.xlabel('sample #')
plt.ylabel('s2')

plt.figure()
plt.scatter(s1, s2)
plt.xlabel('s1')
plt.ylabel('s2')

#%% generate transformed observed data
x1 = (2/3)*s1 + s2
x2 = (2/3)*s1 + (1/3)*s2

X = np.column_stack((x1,x2))

plt.figure()
plt.plot(x1)
plt.xlabel('sample #')
plt.ylabel('x1')

plt.figure()
plt.plot(x2)
plt.xlabel('sample #')
plt.ylabel('x2')

plt.figure()
plt.scatter(x1, x2)
plt.xlabel('x1')
plt.ylabel('x2')
           
#%% extract latent variables via PCA
pca = PCA()
T = pca.fit_transform(X)

plt.figure()
plt.plot(T[:,0])
plt.xlabel('sample #')
plt.ylabel('t1')

plt.figure()
plt.plot(T[:,1])
plt.xlabel('sample #')
plt.ylabel('t2')

plt.figure()
plt.scatter(T[:,0], T[:,1])
plt.xlabel('t1')
plt.ylabel('t2')

#%% extract latent variables via ICA
ica = FastICA()
U = ica.fit_transform(X)

plt.figure()
plt.plot(U[:,0])
plt.xlabel('sample #')
plt.ylabel('u1')

plt.figure()
plt.plot(U[:,1])
plt.xlabel('sample #')
plt.ylabel('u2')

plt.figure()
plt.scatter(U[:,0], U[:,1])
plt.xlabel('u1')
plt.ylabel('u2')

