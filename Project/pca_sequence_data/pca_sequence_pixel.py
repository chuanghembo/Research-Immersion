import torch
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
import time
from sklearn.decomposition import IncrementalPCA, PCA, SparsePCA, KernelPCA
import joblib
import argparse

parser = argparse.ArgumentParser(description='Incremental PCA reduction of input sequence pictures')

parser.add_argument('--infile', type=str, default='./data/HLA_B_4002.txt', help='File containing peptide sequence data')
parser.add_argument('--indir', type=str, default='./data/HLA_B_4002', help='Folder containing peptide sequence images')
parser.add_argument('--n_comp', type=int, default=100, help='Number of iPCA components to keep')
parser.add_argument('--out', type=str, default='incremental_pca_model.pkl', help='Output file for the iPCA model')

argparse = parser.parse_args()

seq_file = argparse.infile
seq_dir = argparse.indir
out_file = argparse.out
n_comp = argparse.n_comp

## Load sequence data
with open(seq_file, 'r') as f:
    peptides = [line.split()[0] for line in f]

## Define function for load peptide sequence images
def load_image(img_path):
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform(img)

## Create peptide dataset class
class PeptideDataset(Dataset):
    def __init__(self, peptides, image_folder):
        self.image_folder = image_folder
        self.peptides = peptides

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        peptide = self.peptides[idx]
        img_path = f'{self.image_folder}/{peptide}.png'
        img_tensor = load_image(img_path)
        return img_tensor.flatten()
    
dataset = PeptideDataset(peptides, seq_dir)
dataset = torch.stack([dataset[i] for i in range(len(dataset))])

t1 = time.time()
# ipca = IncrementalPCA(n_components=n_comp, batch_size=10)
pca = PCA(n_components=n_comp, random_state=42)
pca.fit(dataset)
print(f'Elapsed time: {time.time() - t1}')

joblib.dump(pca, out_file)

