'''

This script generates a dataset of protein sequence pictures 
with protein sequences and raw amino acid pictures as inputs

'''
# %%
## Load in libraries
from PIL import Image
import argparse

# %%
## Get input arguments

parser = argparse.ArgumentParser(description='Generate protein sequence pictures')
parser.add_argument('--infile', type=str, help='Path to file with protein sequences')
parser.add_argument('--indir', type=str, help='Path to directory with amino acid pictures')
parser.add_argument('--outdir', type=str, help='Path to output directory')

args = parser.parse_args()

infile = args.infile
aa_pic_dir = args.indir
outdir = args.outdir 

# %%
amino_acid_full_names = {
    'A': 'alanine', 'R': 'arginine', 'N': 'asparagine', 'D': 'aspartic_acid', 'C': 'cysteine', 
    'E': 'glutamic_acid', 'Q': 'glutamine', 'G': 'glycine', 'H': 'histidine', 'I': 'isoleucine', 
    'L': 'leucine', 'K': 'lysine', 'M': 'methionine', 'F': 'phenylalanine', 'P': 'proline',
    'S': 'serine', 'T': 'threonine', 'W': 'tryptophan', 'Y': 'tyrosine', 'V': 'valine'
}

# %%
## Load in Amino acid depictions
amino_acid_dict = {}

for letter, aa in amino_acid_full_names.items():
    img_path = f'{aa_pic_dir}/{aa}.png'
    image = Image.open(img_path).convert('RGB')
    amino_acid_dict[letter] = image

width, height = amino_acid_dict['A'].size

# %% 
# ## Define function to concatnate aminoacid picture for protein sequence

def concat_pictures(amino_acid_sequence, amino_acid_dict):
    
    new_img = Image.new('RGB', (len(amino_acid_sequence)*width, height))
    
    i = 0
    for aa in amino_acid_sequence:
        im = amino_acid_dict[aa]
        new_img.paste(im, (i,0))
        i += width
         
    return new_img

# %%
## Generate protein sequence pictures

f = open(infile, 'r')
proteins = f.read().split('\n')
f.close()

for protein in proteins:
    protein = protein.upper()[:-2]
    img = concat_pictures(protein, amino_acid_dict)
    img.save(f'{outdir}/{protein}.png')

