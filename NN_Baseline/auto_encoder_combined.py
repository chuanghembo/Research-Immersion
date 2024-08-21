import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the images
depictions_dir = '../data/2Dstruc/'
amino_acid_full_names = {
    'A': 'alanine', 'R': 'arginine', 'N': 'asparagine', 'D': 'aspartic_acid', 'C': 'cysteine', 
    'E': 'glutamic_acid', 'Q': 'glutamine', 'G': 'glycine', 'H': 'histidine', 'I': 'isoleucine', 
    'L': 'leucine', 'K': 'lysine', 'M': 'methionine', 'F': 'phenylalanine', 'P': 'proline',
    'S': 'serine', 'T': 'threonine', 'W': 'tryptophan', 'Y': 'tyrosine', 'V': 'valine'
}

def load_AA_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resizing to a smaller size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path).convert('RGB')
    processed_image = transform(image).unsqueeze(0).to(device)
    return processed_image


# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(27, 64, kernel_size=3, stride=2, padding=1),  # 27 input channels
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=7)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=7),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 27, kernel_size=3, stride=2, padding=1, output_padding=1),  # 27 output channels
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


# Define the Linear Neural Network
class Linear_NN(nn.Module):
    def __init__(self, input_size):
        super(Linear_NN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 16)
        self.fc3 = torch.nn.Linear(16, 1)
        self.drop = torch.nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        out = self.fc3(x)
        return out

# Define the Pixel and VGG Features extraction functions
def pixel_features(amino_acid_full_names):
    aa_features_dict = {}
    for letter, aa in amino_acid_full_names.items():
        img_path = depictions_dir + f'{aa}.png'
        image = load_AA_image(img_path)
        aa_features_dict[letter] = image
    return aa_features_dict


def vgg_features(amino_acid_full_names):
    vgg16_bn = models.vgg16_bn(pretrained=True).features.to(device).eval()
    aa_features_dict = {}
    for letter, aa in amino_acid_full_names.items():
        img_path = depictions_dir + f'{aa}.png'
        image = load_AA_image(img_path)
        with torch.no_grad():
            vgg_feature = vgg16_bn(image).view(-1)
        aa_features_dict[letter] = vgg_feature
    return aa_features_dict



# Load the data (peptides and targets)
data_dir = '../data/'
peptides_file = data_dir + "pan_prediction.txt"
peptides_list = np.loadtxt(peptides_file, dtype=str).tolist()
peptides_list = [[row[2], row[11]] for row in peptides_list]

targets = []
peptides = []
for peptide, score in peptides_list:
    peptides.append(peptide)
    score = float(score)
    targets.append(float(score))

# Split the data into training and evaluation sets
peptides_train, peptides_eval, targets_train, targets_eval = train_test_split(
    peptides, targets, test_size=0.2, random_state=42)

# Prepare the training and evaluation data
def prepare_data(peptides, aa_features_dict):
    images = []
    for peptide in peptides:
        img_list = [aa_features_dict[aa] for aa in peptide]  # List of [3, 64, 64] tensors
        img_tensor = torch.cat(img_list, dim=1)  # Concatenate along the channel dimension (dim=0) to get [27, 64, 64]
        images.append(img_tensor)
    return torch.stack(images).squeeze(1)  # Stack and remove the singleton dimension



# Train the Autoencoder and Linear Neural Network together
def train_combined_model(encoding_method, train_dataset, eval_dataset, epochs=100, batch_size=32):
    autoencoder = Autoencoder().to(device)
    linear_nn = Linear_NN(input_size=25600).to(device)  # 64 comes from the Autoencoder's latent space size

    # Optimizer and loss function
    optimizer = optim.Adam(list(autoencoder.parameters()) + list(linear_nn.parameters()), lr=0.001)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        autoencoder.train()
        linear_nn.train()
        train_loss = 0.0

        for images, targets in train_loader:
            optimizer.zero_grad()

            # Forward pass through autoencoder and linear model
            latent, reconstructed = autoencoder(images)
            outputs = linear_nn(latent.view(latent.size(0), -1))

            # Compute the loss (both reconstruction and prediction)
            reconstruction_loss = criterion(reconstructed, images)
            prediction_loss = criterion(outputs, targets)
            loss = reconstruction_loss + prediction_loss

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        training_losses.append(train_loss / len(train_loader))

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}')

        # Evaluate after each epoch
        autoencoder.eval()
        linear_nn.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for images, targets in eval_loader:
                latent, _ = autoencoder(images)
                outputs = linear_nn(latent.view(latent.size(0), -1))
                loss = criterion(outputs, targets)
                eval_loss += loss.item()

        validation_losses.append(eval_loss / len(eval_loader))

        print(f'Validation Loss: {eval_loss/len(eval_loader):.4f}')

    # Save the models
    torch.save(autoencoder.state_dict(), f'./autoencoder_{encoding_method}.pth')
    torch.save(linear_nn.state_dict(), f'./linear_nn_{encoding_method}.pth')

    # Plot training and validation loss
    plt.figure()
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.legend()
    plt.yscale('log')
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {encoding_method}')
    plt.savefig(f'./loss/training_validation_loss_{encoding_method}.png')
    plt.close()

    print(f"Models saved for {encoding_method}!")

# Evaluation Function
def evaluate_model(encoding_method, models, eval_dataset):
    predictions = []

    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for images, targets in eval_loader:
            images = images.to(device)
            model_predictions = []

            for model in models:
                model.eval()
                latent, _ = autoencoder(images)
                outputs = model(latent.view(latent.size(0), -1))
                model_predictions.extend(outputs.cpu().numpy())

            predictions.append(model_predictions)

    # Average predictions across models
    averaged_predictions = np.mean(predictions, axis=0)

    # Calculate total loss
    total_loss = criterion(torch.tensor(averaged_predictions, dtype=torch.float32).to(device), eval_targets).item()

    return averaged_predictions, total_loss

# Main loop over encoding methods
for encoding_method in ['pixel', 'vgg']:
    if encoding_method == 'pixel':
        aa_features_dict = pixel_features(amino_acid_full_names)
    elif encoding_method == 'vgg':
        aa_features_dict = vgg_features(amino_acid_full_names)

    train_images = prepare_data(peptides_train, aa_features_dict)
    eval_images = prepare_data(peptides_eval, aa_features_dict)

    train_targets_tensor = torch.tensor(targets_train, dtype=torch.float32).to(device).view(-1, 1)
    eval_targets_tensor = torch.tensor(targets_eval, dtype=torch.float32).to(device).view(-1, 1)

    train_dataset = TensorDataset(train_images, train_targets_tensor)
    eval_dataset = TensorDataset(eval_images, eval_targets_tensor)

    # Train the combined model
    train_combined_model(encoding_method, train_dataset, eval_dataset, epochs=100)

    # Load the saved models
    autoencoder = Autoencoder().to(device)
    linear_nn = Linear_NN(input_size=25600).to(device)

    autoencoder.load_state_dict(torch.load(f'./autoencoder_{encoding_method}.pth'))
    linear_nn.load_state_dict(torch.load(f'./linear_nn_{encoding_method}.pth'))

    models = [linear_nn]

    # Test the models on evaluation data
    predictions, total_loss = evaluate_model(encoding_method, models, eval_dataset)

    print(f'Total Evaluation Loss for {encoding_method}: {total_loss:.4f}')

    # Save evaluation results
    outfile = f'./evaluation_result/{encoding_method}_evaluation_predictions.txt'

    with open(outfile, 'w') as f:
        print('Peptide      Score      Prediction', file=f)
        for peptide, score, prediction in zip(peptides_eval, targets_eval, predictions):
            print(f'{"".join(peptide):<12} {score:<10.4f} {prediction[0]:<10.4f}', file=f)

    # Plot evaluation results
    pcc = pearsonr(targets_eval, np.array(predictions).flatten())
    print(f"PCC for {encoding_method}: ", pcc[0])

    plt.figure()
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.scatter(targets_eval, predictions, edgecolors='black')
    plt.xlabel('Target')
    plt.ylabel('Predicted')
    plt.title(f'{encoding_method} - PCC: {pcc[0]:.4f}, Total Loss: {total_loss:.4f}')
    plt.savefig(f'./evaluation_result/evaluation_scatter_{encoding_method}.png')
    plt.close()
