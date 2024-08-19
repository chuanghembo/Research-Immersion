# %% [markdown]
# ## Load Python Libraries

# %%
import torch
from torchvision import transforms
import torchvision.models as models
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset, SubsetRandomSampler


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


from scipy.stats import pearsonr
from PIL import Image

# %% [markdown]
# ## Defining the model

# %%
# Define model
class Linear_NN(torch.nn.Module):

    # 180 is given from the one-hot encoding of the 20 amino acids * 9 peptide length
    def __init__(self, input_size):
        super(Linear_NN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, input_size // 2)
        self.fc2 = torch.nn.Linear(input_size//2, input_size // 10)
        self.fc3 = torch.nn.Linear(input_size//10, 1)
        self.drop = torch.nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.drop(x)
        out = self.fc3(x)
        return out

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# ## Get a list of peptides

# %%
data_dir = '../data/'
peptides_file = data_dir + "binding_affinity.txt"
depictions_dir = data_dir + "2Dstruc/"
peptides_list = np.loadtxt(peptides_file, dtype=str).tolist()
peptides_list = [row[1:] for row in peptides_list if len(row[1]) == 9]
targets = []
peptides = []
for peptide, score in peptides_list:
    peptides.append(peptide)
    score = float(score)
    targets.append(float(score))

# Convert lists to numpy arrays (optional, but often useful)
peptides = np.array(peptides)
targets = np.array(targets)

# Split the data into training and evaluation sets
peptides_train, peptides_eval, targets_train, targets_eval = train_test_split(
    peptides, 
    targets, 
    test_size=0.2, 
    random_state=42
)

# %% [markdown]
# ## One-hot encoding of the pepetides

# %%
def one_hot_encode_peptides(peptides):
    # Convert each peptide sequence into a list of characters
    flattened_peptides = [list(peptide) for peptide in peptides]

    # Flatten the list of lists into a 2D array where each row is a single amino acid
    flattened_peptides = np.array(flattened_peptides).flatten().reshape(-1, 1)

    # Initialize the OneHotEncoder with the 20 standard amino acids
    encoder = OneHotEncoder(categories=[list('ACDEFGHIKLMNPQRSTVWY')], sparse_output=False)

    # Transform the peptide sequences into a one-hot encoded format
    one_hot_encoded = encoder.fit_transform(flattened_peptides)

    # Reshape into the desired format (num_peptides x length_of_each_peptide * 20)
    num_peptides = len(peptides)
    peptide_length = len(peptides[0])
    one_hot_encoded_peptides = one_hot_encoded.reshape(num_peptides, peptide_length * 20)

    return one_hot_encoded_peptides

# %% [markdown]
# ## PCA-reduced Pixel encoding

# %%
amino_acid_full_names = {
    'A': 'alanine', 'R': 'arginine', 'N': 'asparagine', 'D': 'aspartic_acid', 'C': 'cysteine', 
    'E': 'glutamic_acid', 'Q': 'glutamine', 'G': 'glycine', 'H': 'histidine', 'I': 'isoleucine', 
    'L': 'leucine', 'K': 'lysine', 'M': 'methionine', 'F': 'phenylalanine', 'P': 'proline',
    'S': 'serine', 'T': 'threonine', 'W': 'tryptophan', 'Y': 'tyrosine', 'V': 'valine'
}

# %% [markdown]
# ### Load amino acid depictions

# %%
#store images in cache to save performance
image_cache = {}
# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_AA_image(img_path):

    if img_path in image_cache:
        return image_cache[img_path]
    
    # Define transformation to do on image
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    image = Image.open(img_path).convert('RGB')
    processed_image = transform(image).unsqueeze(0).to(device)
    
    image_cache[img_path] = processed_image
    
    return processed_image

# %% [markdown]
# ## PCA pixel feature extraction

# %%
def PCA_pixel_features(amino_acid_full_names):
    
    pixel_features = []

    for letter, aa in amino_acid_full_names.items():
        # load and preprocess
        img_path = depictions_dir + f'{aa}.png'
        image = load_AA_image(img_path)
        pixel_features.append(image.flatten())

    pixel_features = np.vstack(pixel_features)

    # PCA
    pca = PCA(random_state=42)
    pca_pixel_features = pca.fit_transform(pixel_features)

    # Create dictionary with aa_name:PCA_feature
    aa_features_dict = {}
    for idx, aa in enumerate(amino_acid_full_names.keys()):
        aa_features_dict[aa] = pca_pixel_features[idx, :]

    return aa_features_dict, pca

# %%
def pixel_features(amino_acid_full_names):
    
    pixel_features = []

    for letter, aa in amino_acid_full_names.items():
        # load and preprocess
        img_path = depictions_dir + f'{aa}.png'
        image = load_AA_image(img_path)
        pixel_features.append(image.flatten())

    pixel_features = np.vstack(pixel_features)


    # Create dictionary with aa_name:PCA_feature
    aa_features_dict = {}
    for idx, aa in enumerate(amino_acid_full_names.keys()):
        aa_features_dict[aa] = pixel_features[idx, :]

    return aa_features_dict

# %%
def plot_PCA_variance(pca, encoding_method, cumulative=True):
    # Assuming you have already performed PCA and stored it in the variable 'pca'
    variance_ratio = pca.explained_variance_ratio_

    if cumulative:
        cumulative_variance = np.cumsum(variance_ratio)
        plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, '*-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance')
        plt.title('Cumulative Variance by Number of Components, Encoding Method: ' + encoding_method)
        plt.show()
    else:
        plt.bar(range(1, len(variance_ratio)+1), variance_ratio)
        plt.xlabel('Number of Components')
        plt.ylabel('Variance Ratio')
        plt.title('Variance Ratio by Number of Components, Encoding Method: ' + encoding_method)
        plt.show()
    

# %% [markdown]
# ## VGG16 transformation

# %%
def PCA_vgg_features(amino_acid_full_names):

    # Define VGG model
    vgg16_bn = torch.nn.Sequential(   
    # Use only the convolutionary part
    models.vgg16_bn(pretrained = True).features,
    torch.nn.Flatten()
    )

    vgg16 = vgg16_bn.to(device)
    vgg16.eval()

    vgg_features = []

    for aa in amino_acid_full_names.values():
        # load and preprocess
        img_path = img_path = depictions_dir + f'{aa}.png'
        image = load_AA_image(img_path)

        # Disable gradient calculation
        with torch.no_grad():
            conv_features = vgg16(image)

        vgg_features.append(conv_features.cpu().numpy())

    vgg_features = np.vstack(vgg_features)

    pca = PCA(random_state=42)
    pca_vgg_features = pca.fit_transform(vgg_features)

    # Create dictionary with aa_name:PCA_feature
    aa_features_dict = {}
    for idx, aa in enumerate(amino_acid_full_names.keys()):
        aa_features_dict[aa] = pca_vgg_features[idx, :]

    return aa_features_dict, pca

# %%
def vgg_features(amino_acid_full_names):

    # Define VGG model
    vgg16_bn = torch.nn.Sequential(   
    # Use only the convolutionary part
    models.vgg16_bn(pretrained = True).features,
    torch.nn.Flatten()
    )

    vgg16 = vgg16_bn.to(device)
    vgg16.eval()

    vgg_features = []

    for aa in amino_acid_full_names.values():
        # load and preprocess
        img_path = img_path = depictions_dir + f'{aa}.png'
        image = load_AA_image(img_path)

        # Disable gradient calculation
        with torch.no_grad():
            conv_features = vgg16(image)

        vgg_features.append(conv_features.cpu().numpy())

    vgg_features = np.vstack(vgg_features)

    # Create dictionary with aa_name:PCA_feature
    aa_features_dict = {}
    for idx, aa in enumerate(amino_acid_full_names.keys()):
        aa_features_dict[aa] = vgg_features[idx, :]

    return aa_features_dict

# %% [markdown]
# ## Peptide data encoding

# %%
def Encoder(peptides, aa_features_dict):
    
    encoded_peptides = []

    for peptide in peptides:
        encoded_peptide = []
    
        for aa in peptide:
            encoded_peptide.append(aa_features_dict[aa])
        
        encoded_peptide = np.array(encoded_peptide).flatten()
        encoded_peptides.append(encoded_peptide)

    return np.array(encoded_peptides)

# %%

for encoding_method in ['onehot', 'pca', 'vgg', 'pixel', 'vgg_features']:
    if encoding_method == 'onehot':
        encoded_peptides = one_hot_encode_peptides(peptides_train)

    elif encoding_method == 'pca':
        aa_feature_dict, pca = PCA_pixel_features(amino_acid_full_names)
        plot_PCA_variance(pca, encoding_method, cumulative=True)
        encoded_peptides = Encoder(peptides_train, aa_feature_dict)

    elif encoding_method == 'vgg':
        aa_feature_dict, pca = PCA_vgg_features(amino_acid_full_names)
        plot_PCA_variance(pca, encoding_method, cumulative=True)
        encoded_peptides = Encoder(peptides_train, aa_feature_dict)

    elif encoding_method == 'pixel':
        aa_feature_dict = pixel_features(amino_acid_full_names)
        encoded_peptides = Encoder(peptides_train, aa_feature_dict)

    elif encoding_method == 'vgg_features':
        aa_feature_dict = vgg_features(amino_acid_full_names)
        encoded_peptides = Encoder(peptides_train, aa_feature_dict)

    else:
        raise ValueError('Invalid encoding method')


    input_size = encoded_peptides[0].shape[0]

    # %% [markdown]
    # ## Define function to reset weight
    # Weight resetting aid to prevent the weight leakage

    # %%
    def reset_weights(m):
        if hasattr(m, 'reset_parameters'):
            print(f'reset weight of layer {m}')
            m.reset_parameters()

    # %% [markdown]
    # ### Loss function and optimizer

    # %%
    # Define the loss function 
    criterion = torch.nn.MSELoss()

    # %%
    peptides_tensor = torch.tensor(encoded_peptides, dtype=torch.float32).to(device)
    target_tensor = torch.tensor(np.asarray(targets_train).reshape(-1,1), dtype=torch.float32).to(device)

    peptides_dataset = TensorDataset(peptides_tensor, target_tensor)


    # %% [markdown]
    # ## Train model

    # %%
    def plot_loss(train_loss, val_loss, fold, encoding_method):
        plt.figure()
        plt.plot(train_loss, label='train err')
        plt.plot(val_loss, label='val err')
        plt.legend()
        plt.yscale('log')
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'./loss/{encoding_method}_loss_fold_{fold}.png')
        plt.close()

    # %%
    torch.manual_seed(69)

    kfold = KFold(n_splits=5, shuffle=True)

    print("Starting KFold Cross Validation")

    for fold, (train_ids, test_ids) in enumerate(kfold.split(peptides_dataset), 1):
        
        print(f'Fold {fold}')

        # Shuffle the data
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        # Define the data loaders
        train_loader = DataLoader(peptides_dataset, batch_size=10, sampler=train_subsampler)
        test_loader = DataLoader(peptides_dataset, batch_size=10, sampler=test_subsampler)

        # Initialize NN
        model = Linear_NN(input_size).to(device)
        model.apply(reset_weights)

        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        training_losses = []
        validation_losses = []

        for epoch in range(0, 10):
        
                print(f'Epoch {epoch+1}')

                training_loss = 0.0

                for i, data in enumerate(train_loader):
                    
                    # Get the inputs
                    inputs, labels = data

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs.view(-1, encoded_peptides.shape[1]))

                    # Calculate the loss
                    loss = criterion(outputs, labels)

                    # Backward pass
                    loss.backward()

                    # Optimize
                    optimizer.step()

                    training_loss += loss.item()
                
                with torch.no_grad():
                    model.eval()
                    validation_loss = 0.0

                    for inputs, labels in test_loader:

                        outputs = model(inputs.view(-1, encoded_peptides.shape[1]))
                        loss = criterion(outputs, labels)

                        validation_loss += loss.item()

                training_losses.append(training_loss/len(train_loader))
                validation_losses.append(validation_loss/len(test_loader))

                print(f'Training loss: {training_loss/len(train_loader)}')
                print(f'Validation Loss: {validation_loss/len(test_loader)}')

                
        # Save the model
        print('Finished Training, Saving Model')
        torch.save(model.state_dict(), f'./model/{encoding_method}_model_fold_{fold}.pt')

        # Plot the loss
        plot_loss(training_losses, validation_losses, fold, encoding_method) 
    
        

    # %% [markdown]
    # ## Evaluating the model with unseen data

    # %%
    if encoding_method == 'onehot':
        encoded_evaluation_peptides = one_hot_encode_peptides(peptides_eval)
    else:
        encoded_evaluation_peptides = Encoder(peptides_eval, aa_feature_dict)

    # Convert to tensors
    evaluation_peptides_tensor = torch.tensor(encoded_evaluation_peptides, dtype=torch.float32).to(device)
    evaluation_score_tensor = torch.tensor(np.asarray(targets_eval).reshape(-1,1), dtype=torch.float32).to(device)

    evaluation_peptides_dataset = TensorDataset(evaluation_peptides_tensor, evaluation_score_tensor)

    def evaluate_model(models, evaluation_peptides_dataset):
        predictions = []

        evaluation_data_loader = DataLoader(evaluation_peptides_dataset, batch_size=10)

        with torch.no_grad():
            for model in models:
                model.eval()
                model_predictions = []

                for peptides, _ in evaluation_data_loader:
                    peptides = peptides.to(device)
                    outputs = model(peptides.view(-1, 180))
                    model_predictions.extend(outputs.cpu().numpy())

                predictions.append(model_predictions)

        # Average predictions across models
        averaged_predictions = np.mean(predictions, axis=0)

        # Calculate total loss
        total_loss = criterion(torch.tensor(averaged_predictions, dtype=torch.float32).to(device), evaluation_score_tensor).item()

        return averaged_predictions, total_loss

    # Load models
    models = []
    for i in range(1,6):
        model = Linear_NN(input_size).to(device)
        model.load_state_dict(torch.load(f'./model/{encoding_method}_model_fold_{i}.pt'))
        models.append(model)

    # Test models
    predictions, total_loss = evaluate_model(models, evaluation_peptides_dataset)

    print(f'Total Loss: {total_loss}')

    # %% [markdown]
    # ## Saving evaluation data prediction result

    # %%
    outfile = f'./evaluation_result/{encoding_method}_evaluation_predictions.txt'

    with open(outfile, 'w') as f:
        print('Peptide      Score      Prediction', file=f)
        for peptide, score, prediction in zip(peptides_eval, targets_eval, predictions):
            print(f'{"".join(peptide):<12} {score:<10.4f} {prediction[0]:<10.4f}', file=f)

    # %%
    pcc = pearsonr(targets_eval, np.array(predictions).flatten())
    print("PCC: ", pcc[0])

    plt.figure()
    plt.style.use('seaborn-v0_8-whitegrid');
    plt.scatter(targets_eval, predictions, edgecolors='black');
    plt.xlabel('Target PSSM');
    plt.ylabel('Predicted PSSM');
    plt.title(f'Encoding Method: {encoding_method}, PCC: {pcc[0]:.4f}, total loss: {total_loss:.4f}');
    plt.savefig(f'./evaluation_result/{encoding_method}_evaluation_scatter.png')


