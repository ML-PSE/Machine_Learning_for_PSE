##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          Illustration example for FDA/LDA
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#%% generate data
x1_class1 = np.random.uniform(1, 6, 100)
x2_class1 = x1_class1 + 1 + np.random.normal(0,0.5,100)
X_class1 = np.column_stack((x1_class1, x2_class1))


x1_class2 = np.random.uniform(2, 7, 100)
x2_class2 = x1_class2 - 1 + np.random.normal(0,0.5,100)
X_class2 = np.column_stack((x1_class2, x2_class2))

plt.figure()
plt.plot(x1_class1, x2_class1, 'b.', label='Class 1')
plt.plot(x1_class2, x2_class2, 'r.', label='Class 2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

X = np.vstack((X_class1, X_class2))
y = np.concatenate((np.ones(100,), 2*np.ones(100,)))

#%% scale data
scalar = StandardScaler()
X_normal = scalar.fit_transform(X)
           
#%% extract latent variables via PCA
pca = PCA(n_components=1)
score_pca = pca.fit_transform(X_normal)

plt.figure()
plt.plot(score_pca[0:100], np.zeros((100,)), 'b.')
plt.plot(score_pca[100:], np.zeros((100,)), 'r.')
plt.ylim((-2,100))
plt.xlabel('PCA score')
plt.ylabel('sample #')

#%% extract latent variables via LDA
lda = LinearDiscriminantAnalysis(n_components=1)
score_lda = lda.fit_transform(X_normal, y)

plt.figure()
plt.plot(score_lda[0:100], np.zeros((100,)), 'b.')
plt.plot(score_lda[100:], np.zeros((100,)), 'r.')
plt.ylim((-2,100))
plt.xlabel('LDA score')
plt.ylabel('sample #')