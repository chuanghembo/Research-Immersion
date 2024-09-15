import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

parser = argparse.ArgumentParser(description='Autoencoder training')

parser.add_argument('-infile', type=str, default='../0_data/HLA_B_4002_train.txt', help='File containing peptide sequence data')
parser.add_argument('-indir', type=str, default='../0_data/seq_pic/', help='Folder containing peptide sequence images')
parser.add_argument('--out', type=str, default='autoencoder.pth', help='Output file for the trained model')
parser.add_argument('--op', type=str, default='./', help='Output location of the reconstruction pictures')

args = parser.parse_args()

infile = args.infile
indir = args.indir
out = args.out
op = args.op


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
def set_seed(seed):
    random.seed(seed)            
    np.random.seed(seed)         
    torch.manual_seed(seed)      
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(49)


prot_seq_f = open(infile, 'r')
prot_seq = prot_seq_f.read().split('\n')
prot_seq_f.close()
prot_seq = [peptide for peptide in prot_seq if peptide]

for i in range(10):
	print(prot_seq[i])



prot_pics = []

for seq in prot_seq:
    transform = transforms.ToTensor()
    image = Image.open(f'{indir}{seq}.png')
    image = transform(image)
    prot_pics.append(image)

train, test = train_test_split(prot_pics, test_size=0.2)
print(f'number of training samples: {len(train)}')
print(f'number of testing samples: {len(test)}')

train_loader = DataLoader(train, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=4)

RANDOM_SEED = 49
LEARNING_RATE = 0.0005
BATCH_SIZE = 256
NUM_EPOCHS = 10

print('Training Set:\n')
for images in train_loader:
    print('Image batch dimensions:', images.size())
    # print('Image label dimensions:', labels.size())
    # print(labels[:10])
    break

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    def __init__(self, target_height, target_width):
        super(Trim, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, x):
        return x[:, :, :self.target_height, :self.target_width]



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),  # (224, 2016) -> (224, 2016)
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),  # (224, 2016) -> (112, 1008)
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),  # (112, 1008) -> (56, 504)
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),  # (56, 504) -> (56, 504)
            nn.Flatten(),
            nn.Linear(64 * 56 * 504, 2016)  # Adjusted Linear layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(2016, 64 * 56 * 504),
            Reshape(-1, 64, 56, 504),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1, output_padding=(1, 1)),  # (56, 504) -> (112, 1008)
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=1, output_padding=(1, 1)),  # (112, 1008) -> (224, 2016)
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 3, stride=(1, 1), kernel_size=(3, 3), padding=1),  # (224, 2016) -> (224, 2016)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = AutoEncoder()
x = torch.randn(1, 3, 224, 224*9)  # Example input tensor with size (batch_size, channels, height, width)
output = model(x)
print(output.shape)

model = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def compute_epoch_loss_autoencoder(model, dataloader, loss_fn):
    
    loss = 0.0 
    predictions = []

    for features in dataloader:
        features = features.to(device)
        logits = model(features)
        loss += loss_fn(logits, features)
        predictions.extend(logits)

    return loss / len(dataloader), predictions



def train_model(num_epochs, model, optimizer, train_loader, loss_fn=None,
                logging_interval=100, skip_epoch_stats = False, save_model = None):
    
    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}

    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, features in enumerate(train_loader):

            features = features.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            print(logits.shape)
            loss = loss_fn(logits, features)
            optimizer.zero_grad()
            loss.backward()
            # UPDATE MODEL PARAMETERS
            optimizer.step()
            
            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx, len(train_loader), loss))
       
        if not skip_epoch_stats:
            model.eval()
            with torch.no_grad():  # save memory during inference
                
                train_loss = compute_epoch_loss_autoencoder(model, train_loader, loss_fn)

                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())


    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    return log_dict

log_dict = train_model(num_epochs=NUM_EPOCHS, model=model,
                                optimizer=optimizer,
                                train_loader=train_loader,
                                skip_epoch_stats=True,
                                logging_interval=250,
                                save_model=out)

model = AutoEncoder().to(device)
model.load_state_dict(torch.load(out))
model.eval()
with torch.no_grad():
    err, recon = compute_epoch_loss_autoencoder(model, test_loader, nn.MSELoss())


def plot_image(data, recon, num_images, p):
    plt.figure(figsize=(18, 4))

    for i in range(num_images):
        index = p * 2 + i  # Adjust index to increase by 2 every time
        plt.subplot(2, num_images, i+1)
        plt.imshow(data[index].cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')

        plt.subplot(2, num_images, num_images+i+1)
        plt.imshow(recon[index].cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')

    # Save the figure instead of showing it
    plt.savefig(f'{op}output_image_{p}.png')
    plt.close()  # Close the figure to free up memory

for p in range(10):
    plot_image(prot_pics, recon, 2, p)
