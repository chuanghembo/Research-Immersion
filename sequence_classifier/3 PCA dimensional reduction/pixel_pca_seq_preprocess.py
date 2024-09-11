
'''

Creats pytorch dataset of PCA-redcued pixel features from peptide sequence images

'''

# %%
## Load in libraries

import torch
from torchvision import transforms

import numpy as np

from PIL import Image

import argparse
import joblib

# %%
# Get input arguments

parser = argparse.ArgumentParser(description='Construction of PCA Pixel dataset from peptide sequence picture data')

parser.add_argument('--infile', type=str, default='./data/HLA_B_4002_train.txt', help='File containing peptide sequence data')
parser.add_argument('--indir', type=str, default='./data/HLA_B_4002', help='Folder containing peptide sequence images')
parser.add_argument('--model', type=str, default='pixel_pca_seq_model.pkl', help='Output file for the PCA model')
parser.add_argument('--out', type=str, default='pixel_pca_dataset.pt', help='Output file for the dataset')

args = parser.parse_args()

seq_file = args.infile
seq_dir = args.indir
pca_model_file = args.model
out_file = args.out

# %%
# Load sequence data

with open(seq_file, 'r') as f:
    peptides = {line.split()[0]: float(line.split()[1]) for line in f}

# %%
# Define image loader

def load_image(img_path):
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(img)

# %%
# Load PCA model
pca_model = joblib.load(pca_model_file)

# %%
# Prepare lists for storing features and labels
features_list = []
labels_list = []
peptides_list = []

# %%
# Process each image and apply PCA
for peptide, label in peptides.items():
    img_path = f'{seq_dir}/{peptide}.png'
    img_tensor = load_image(img_path)
    
    # Flatten and apply PCA
    img_flat = img_tensor.view(-1).numpy().reshape(1, -1)
    img_pca = pca_model.transform(img_flat)
    
    # Convert back to torch tensor
    img_pca_tensor = torch.tensor(img_pca, dtype=torch.float32).squeeze()
    
    features_list.append(img_pca_tensor)
    labels_list.append(label)
    peptides_list.append(peptide)

# %%
# Convert lists to tensors
features_tensor = torch.stack(features_list)
labels_tensor = torch.tensor(labels_list, dtype=torch.float32)

# %%
# Save the features and labels as .pt file 
torch.save({'peptides':peptides_list, 'features': features_tensor, 'labels': labels_tensor}, out_file)