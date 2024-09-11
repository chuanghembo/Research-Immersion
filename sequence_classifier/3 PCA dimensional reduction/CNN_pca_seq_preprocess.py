'''

Generate pytorch dataset of PCA-reduced CNN features from peptide sequence images

'''

# %%
## Load in libraries

import torch
import numpy as np

import argparse
import joblib

# %%
# Get input arguments

parser = argparse.ArgumentParser(description='Construction of PCA CNN dataset from peptide sequence picture data')

parser.add_argument('--infile', type=str, default='./data/HLA_B_4002_train.txt', help='File containing peptide sequence data')
parser.add_argument('--indir', type=str, default='./data/HLA_B_4002', help='Folder containing peptide sequence images')
parser.add_argument('--raw', type=str, help='File containing raw features from CNN')
parser.add_argument('--model', type=str, default='pca_pixel_seq_model.pkl', help='Output file for the PCA model')
parser.add_argument('--out', type=str, default='pixel_pca_dataset.pth', help='Output file for the dataset')

args = parser.parse_args()

seq_file = args.infile
seq_dir = args.indir
raw = args.raw
pca_model_file = args.model
out_file = args.out

# %%
# Load sequence data

with open(seq_file, 'r') as f:
    peptides = {line.split()[0]: float(line.split()[1]) for line in f}

# %%
# Load raw features
raw_features = joblib.load(raw)

# %%
# Load PCA model
pca_model = joblib.load(pca_model_file)

# %%
features_list = []
labels_list = []
peptides_list = []

# Process each image and apply PCA
for peptide, label in peptides.items():
    
    # Flatten and apply PCA
    features_flat = raw_features[peptide].reshape(1, -1) 
    features_pca = pca_model.transform(features_flat)
    
    features_pca_tensor = torch.tensor(features_pca, dtype=torch.float32).squeeze()
    
    features_list.append(features_pca_tensor)
    labels_list.append(label)
    peptides_list.append(peptide)

# %%
# Convert lists to tensors
features_tensor = torch.stack(features_list)
labels_tensor = torch.tensor(labels_list, dtype=torch.float32)

# %%
# Save the features and labels as .pt file 
torch.save({'peptides':peptides_list, 'features': features_tensor, 'labels': labels_tensor}, out_file)