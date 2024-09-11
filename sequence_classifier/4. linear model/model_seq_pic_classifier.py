# %%
## Load in libraries

import torch
from torch import optim
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
# roc_curve and average precision score requires the predicted probability instead of predicted label as input
from sklearn.metrics import accuracy_score, roc_curve, auc, average_precision_score, recall_score, f1_score, precision_score

import tqdm.notebook as tqdm
import copy
import argparse

# %%
## Get input arguments

parser = argparse.ArgumentParser(description='Sequence input model training and evaluation')
parser.add_argument('--train', type=str, default='pixel_pca_train.pt', help='File containing peptide sequence data in .pt format')
parser.add_argument('--test', type=str, default='pixel_pca_test.pt', help='File containing peptide sequence data in .pt format')
parser.add_argument('--encoding', type=str, default='pixel', help='Encoding method for peptide sequence data')

args = parser.parse_args()

train = args.train
test = args.test
encoding_method = args.encoding
RANDOM_SEED = 42
N_SPLITS = 5
N_EPOCHS = 100

# %%
## Define Dataset class

class PCAPeptideMNIST(Dataset):
    def __init__(self, file_path):
        # Load the .pt file containing the features and labels
        data = torch.load(file_path)
        self.features = data['features']
        self.labels = data['labels']
        self.peptides = data['peptides']

    def __len__(self):
        # Return the number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Return a single sample (PCA-transformed image and corresponding label)
        return self.features[idx], self.labels[idx]

# %%
# Load the training and test datasets

train_dataset = torch.load(train)
test_dataset = torch.load(test)

# %%
## Define Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(RANDOM_SEED)

class Linear_NN(torch.nn.Module):

    def __init__(self, input_size):
        super(Linear_NN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, input_size // 2)
        self.fc2 = torch.nn.Linear(input_size//2, input_size // 10)
        self.fc3 = torch.nn.Linear(input_size//10, 1) 
        self.drop = torch.nn.Dropout(p=0.5)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.drop(x)
        out = self.fc3(x)
        out = self.sigmoid(out)
        return out

# %% 
# ## Define function for OneHot Encoding

def one_hot_encode_peptides(peptides):

    flattened_peptides = np.array(peptides).flatten().reshape(-1, 1)
    
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(categories=[list('ACDEFGHIKLMNPQRSTVWY')], sparse_output=False)

    # Transform the peptide sequences into a one-hot encoded format
    one_hot_encoded = encoder.fit_transform(flattened_peptides)

    
    num_peptides = len(peptides)
    
    one_hot_encoded = one_hot_encoded.reshape(num_peptides, -1)
        
    return one_hot_encoded


# %%
# Prepare the data for training

# One-hot encode the peptide sequences (Baseline)
if encoding_method == 'onehot':
    X_train = torch.tensor(one_hot_encode_peptides([list(peptide) for peptide in train_dataset['peptides']]), dtype=torch.float32).to(device)
    y_train = train_dataset['labels'].to(device)

    X_test = torch.tensor(one_hot_encode_peptides([list(peptide) for peptide in test_dataset['peptides']]), dtype=torch.float32).to(device)
    y_test = test_dataset['labels'].to(device)

else:
    X_train = train_dataset['features'].to(device)
    y_train = train_dataset['labels'].to(device)

    X_test = test_dataset['features'].to(device)
    y_test = test_dataset['labels'].to(device)

input_size = X_train.shape[1]

# %%
# Define function to plot loss

def plot_loss(train_loss, val_loss, val_acc, fold):
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))

    for i, ytitle in enumerate(['loss', ['log loss']]):
        ax[i].plot(train_loss, label='train loss', linestyle='-.')
        ax[i].plot(val_loss, label='val loss', linestyle='-.')
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel(ytitle)
        ax[i].legend()
    ax[1].set_yscale('log')

    ax[2].plot(val_acc, label='val acc',  linestyle='-.')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Accuracy')
    ax[2].legend()

    plt.savefig(f'./\5 result/loss/{encoding_method}_loss_fold_{fold}.png')
    plt.close()

# %%
# Define training loop function

def model_train(model, X_train, y_train, X_val, y_val, fold):

    # loss function and optimizer
    loss_fn = torch.nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = N_EPOCHS   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    train_loss = []
    val_loss = []
    val_acc = []

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True, leave=False) as bar:
            bar.set_description(f"Epoch {epoch+1}")
            
            for start in bar:

                # take a batch
                X_batch = X_train[start:start + batch_size]  
                y_batch = y_train[start:start + batch_size]  
                
                # forward pass
                y_pred = model(X_batch).squeeze()
                loss = loss_fn(y_pred, y_batch)
                
                epoch_loss += loss.item()

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        
        train_loss.append(epoch_loss/len(batch_start))

        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val.to(device)).squeeze()  # Ensure validation data is on the same device
        epoch_val_loss = loss_fn(y_pred, y_val.to(device)).item()  # Move validation labels to the correct device
        acc = accuracy_score(y_val.cpu().numpy(), y_pred.round().cpu().detach().numpy())  # Move to CPU for accuracy calculation
        
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        
        val_loss.append(epoch_val_loss)
        val_acc.append(acc)
    
    # Save the best model weights
    torch.save(best_weights, f'./\5 result/model/{encoding_method}_model_fold_{fold}.pt')
    # Plot the loss
    plot_loss(train_loss, val_loss, val_acc, fold)
    
    return best_acc


# %%
# Train using 5-fold cross validation
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_train.cpu(), y_train.cpu()), 1):

    model = Linear_NN(input_size).to(device)

    acc = model_train(model, X_train[train_idx], y_train[train_idx], X_train[test_idx], y_train[test_idx], fold)

    print(f"Accuracy {fold}: %.5f" % acc)
    cv_scores.append(acc)

# %% 
## Define function to plot roc curve

def plot_roc(y_true, y_pred, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')

# %%
# Evaluate the ensemble model

def evaluate_ensembl(X_test, y_test, n_splits):  
    
    perf_f = open(f'./\5 result/evaluation_result/performance/{encoding_method}_performance.txt', 'w')

    metrics = {
        'accuracy': accuracy_score,
        'average precision score': average_precision_score,
        'recall score': recall_score,
        'f1 score': f1_score
    }

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random', color = 'k')

    ensembl_predictions = []
    score_lists = {name: [] for name in metrics.keys()}


    for i in range(n_splits):
        
        model = Linear_NN(input_size).to(device)

        model.load_state_dict(torch.load(f'./\5 result/model/{encoding_method}_model_fold_{i+1}.pt'))

        model.eval()
        with torch.no_grad():
            y_pred_prob = model(X_test).cpu()
            ensembl_predictions.append(y_pred_prob)
        
        # plot roc for each fold
        plot_roc(y_test.cpu(), y_pred_prob, label=f'Fold {i+1}')

        # scoring each fold using the metrics
        thereshold = 0.5
        y_pred = (y_pred_prob.numpy() > thereshold).astype(int)
        for name, scoring in metrics.items():
            score_lists[name].append(scoring(y_test.cpu(), y_pred))


    avg_ensembl_predictions = torch.stack(ensembl_predictions).mean(dim=0)

    # Print score for each fold
    print(f"{'-'*50}\nCross Validation\n{'-'*50}\n", file=perf_f)
    for name, scores in score_lists.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"{name}", file=perf_f)
            print(f"5-Fold Cross-Validation {name} scores: {np.round(scores, 3)}", file=perf_f)
            print(f"Mean {name} score: {mean_score:.3f} +/- {std_score:.3f}\n", file=perf_f)
    
    # Plot roc for ensembl
    plot_roc(y_test.cpu(), avg_ensembl_predictions.cpu(), label='Ensemble')
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.savefig(f'./\5 result/evaluation_result/roc/{encoding_method}_roc.png')
    plt.close()

    # Calculate metrics scores for ensembl
    avg_ensembl_predictions = (avg_ensembl_predictions.round().numpy() > thereshold).astype(int)
    print(f"{'-'*50}\nEnsembl\n{'-'*50}\n", file=perf_f)
    for name, scoring in metrics.items():
        print(f"{name}", file=perf_f)
        print(f"Ensembl {name} score: {scoring(y_test.cpu(),avg_ensembl_predictions):.3f}\n", file=perf_f)

    perf_f.close()

    ensembl_predictions = [predictions.round().numpy() for predictions in ensembl_predictions]

    # Save the prediction results
    pred_f = open(f'./\5 result/evaluation_result/predictions/{encoding_method}_predictions.txt', 'w')
    y_test = y_test.cpu().numpy()
    print('ytest  fold_1  fold_2  fold_3  fold_4  fold_5  ensembl', file = pred_f)
    for i in range(len(y_test)):
        print(y_test[i], end = '\t', file = pred_f)
        
        for j in range(len(ensembl_predictions)):
            print(ensembl_predictions[j][i], end = '\t', file = pred_f)
        
        print(avg_ensembl_predictions[i], file = pred_f)
    pred_f.close()

# %%
evaluate_ensembl(X_test, y_test, N_SPLITS)


