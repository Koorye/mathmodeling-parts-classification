import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

TITLE = 'PCA'
X_LABEL = 'X'
Y_LABEL = 'Y'
COL_list = ['GDP', 'First', 'Second', 'Third']  # The name of matrix header and element of the bar

pd_data = pd.read_csv('data/pca_data.csv', header=None)
pd_data.columns = COL_list
pd.set_option('display.max_columns', None)

data = np.loadtxt('data/pca_data.csv', delimiter=',')
pca = PCA(n_components=data.shape[1])  # Set the number of variance ratio same as number of element
pca.fit(data)

print('Similar Matrix: ')
print(pd_data.corr())  # Calculating similar values
print('Variance Ratio: ')
print(pca.explained_variance_ratio_)

plt.title(TITLE)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.bar(COL_list, pca.explained_variance_ratio_)
plt.savefig('result/PCA.png')
plt.show()
