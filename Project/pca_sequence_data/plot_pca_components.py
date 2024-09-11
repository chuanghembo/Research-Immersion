import numpy as np
import matplotlib.pyplot as plt

import joblib
import argparse

parser = argparse.ArgumentParser(description='Plot PCA components')
parser.add_argument('--model', type=str, help='Input PCA model file')
parser.add_argument('--out', type=str, default='pca_components.png', help='Output file for the plot')

args = parser.parse_args()

model_file = args.model
out_file = args.out

ipca = joblib.load(model_file)

cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
index_95 = np.argmax(cumulative_variance >= 0.95)
index_99 = np.argmax(cumulative_variance >= 0.999)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(cumulative_variance, '-*')
ax[0].axvline(index_95, color='r', linestyle='--')
ax[0].axvline(index_99, color='r', linestyle='--')
ax[0].set_xlabel('Number of components')
ax[0].set_ylabel('Cumulative explained variance')
ax[0].set_title('Cumulative explained variance')

ax[1].plot(ipca.explained_variance_ratio_, '-*')
ax[1].set_xlabel('Number of components')
ax[1].set_ylabel('Explained variance')
ax[1].set_title('Explained variance')

plt.savefig(out_file)
print(f'95% explained variance: {index_95} components and 99.9% explained variance: {index_99} components')
plt.show()
