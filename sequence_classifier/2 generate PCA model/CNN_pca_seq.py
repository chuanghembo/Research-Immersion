'''

This script extracts features from a pre-trained CNN model (VGG16 or ResNet50) and reduces the dimensionality 
of the features using PCA. 

The script need to be run twice once to train a PCA model and second time to extract features from a dataset 
using a pre-trained PCA model.

'''

# %%
## Load in libraries

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision import models
from torchvision.models import VGG16_BN_Weights, ResNet50_Weights

from sklearn.decomposition import PCA
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import joblib
import argparse

# %%
## Get input arguments

parser = argparse.ArgumentParser(description='VGG feature extraction and PCA reduction of sequence pictures')
parser.add_argument('-infile', type = str, default='./data/HLA_B_4002_train.txt', help='File containing peptide sequence data')
parser.add_argument('-indir', type = str, default='./data/HLA_B_4002', help='Folder containing peptide sequence images')
parser.add_argument('--train', action='store_true', help='Train PCA model')
parser.add_argument('--n_comp', type=int, default=100, help='Number of PCA components to keep')
parser.add_argument('--model', type=str, default='vgg', help='CNN used for feature extraction')
parser.add_argument('--out', type=str, default='vgg_feature_dict.joblib', help='Output file for the feature dictionary')

args = parser.parse_args()

N_COMPONENT = args.n_comp
RANDOM_SEED = 42
MODEL = args.model
infile = args.infile
indir = args.indir
out = args.out

# %%
# Get the pepetides and their labels
with open(infile, 'r') as f:
    peptides = [line.split()[0] for line in f]

# %%
## Define function to load image and transform it to tensor

def load_image(img_path):
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.ToTensor(),

        # Normalize the image with the mean and standard deviation of the ImageNet data
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    return transform(img)

# %%
## Define dataset class

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
        return peptide, img_tensor
    
dataset = PeptideDataset(peptides, indir)
# dataset = [dataset[i] for i in range(9)]
dataloader = DataLoader(dataset, batch_size=3, shuffle=False, pin_memory=True)


# %%
## Load the pre-trained model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL == 'vgg':
    # Load the VGG16 model and prepare it for feature extraction
    vgg16 = models.vgg16_bn(weights = VGG16_BN_Weights.DEFAULT)
    model = nn.Sequential(
        vgg16.features,
        nn.Flatten()).to(device)
    
elif MODEL == 'resnet':
    resnet = models.resnet50(weights = ResNet50_Weights.DEFAULT)
    model = nn.Sequential(
        resnet,
        nn.Flatten()).to(device)


# %%

if args.train:

    ## Extract features from the CNN output and reduce the dimensionality using PCA

    # Batch processing
    peptide_features = []
    model.eval()
    with torch.no_grad():
        for lab, imgs in dataloader:
            imgs = imgs.to(device)
            features = model(imgs)  
            peptide_features.append(features.cpu())  

    peptide_features = torch.cat(peptide_features)

    pca = PCA(n_components=N_COMPONENT,random_state=RANDOM_SEED)
    pca.fit(peptide_features)
    
    # Save the PCA model
    joblib.dump(pca, f'{MODEL}_pca_model.pkl')

else:

    ## Extract features from the CNN output

    peptide_dict = {}

    model.eval()
    with torch.no_grad():
        for labs, imgs in dataloader:
            imgs = imgs.to(device)
            features = model(imgs)  # Forward pass

            for i, lab in enumerate(labs):
                    peptide_dict[lab] = features[i].cpu()  # Store the features as CPU tensors

    # Save the feature dictionary
    joblib.dump(peptide_dict, out)

