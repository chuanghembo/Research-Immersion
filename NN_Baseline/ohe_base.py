import sys
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold

import numpy as np

data_dir = '../data/'

#peptides_file = data_dir + "PSSM/A0201.single_lig"
#peptides_file = data_dir + "PSSM/A0201.small_lig"
#peptides_file = data_dir + "PSSM/A0201.large_lig"
peptides_file = data_dir + "PSSM/A0201.eval"

peptides = np.loadtxt(peptides_file, dtype=str).tolist()

#peptides_seq = [peptide[0] for peptide in peptides]

#One-hot encoding
aa = "ACDEFGHIKLMNPQRSTVWY"
aa_to_int = dict((c, i) for i, c in enumerate(aa))


def encode_peptide(peptide):
        encoding = np.zeros((len(peptide), len(aa)))
        for i, AA in enumerate(peptide):
            encoding[i, aa_to_int[AA]] = 1
        return encoding

targets = []
encodings = []
for peptide, score in peptides:
    X = np.array([encode_peptide(amino_acid) for amino_acid in peptide])
    score = float(score)
    encodings.append(X)
    targets.append(float(score))

tensor_input = torch.stack([torch.tensor(arr) for arr in encodings])
tensor_input = tensor_input.squeeze(dim=2).float()  
targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)  

encodings = torch.stack([torch.tensor(arr) for arr in encodings]).squeeze(dim=2).float()
targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

eval_size = int(0.2 * len(encodings))
train_encodings, eval_encodings = encodings[:-eval_size], encodings[-eval_size:]
train_targets, eval_targets = targets[:-eval_size], targets[-eval_size:]

class OHEBase(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(OHEBase, self).__init__()
        self.hidden_size = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.hidden_size(x))
        x = self.output(x)
        return x

input_size = len(encodings[0]) * 20
hidden_size = 16

model = OHEBase(input_size, hidden_size)
loss_function = nn.MSELoss()


torch.manual_seed(69)

outer_kf = KFold(n_splits=5, shuffle=True)
inner_kf = KFold(n_splits=5, shuffle=True)

outer_fold_results = []

for outer_fold, (train_index, val_index) in enumerate(outer_kf.split(train_encodings)):
    print(f"Outer Fold {outer_fold+1}/5")
    
    # Split the data into training and validation sets
    X_train, X_val = train_encodings[train_index], train_encodings[val_index]
    y_train, y_val = train_targets[train_index], train_targets[val_index]
    
    # Inner cross-validation (used here if you want to tune hyperparameters)
    inner_fold_results = []
    
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kf.split(X_train)):
        print(f"  Inner Fold {inner_fold+1}/5")
        
        X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
        y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
        
        # Train the model
        model = OHEBase(input_size, hidden_size)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_inner_train)
            loss = loss_function(outputs, y_inner_train)
            loss.backward()
            optimizer.step()
        
        # Evaluate the model on the inner validation set
        with torch.no_grad():
            val_outputs = model(X_inner_val)
            val_loss = loss_function(val_outputs, y_inner_val).item()
            inner_fold_results.append(val_loss)
    
    # Average validation loss over the inner folds
    avg_inner_val_loss = np.mean(inner_fold_results)
    print(f"  Average Inner Validation Loss: {avg_inner_val_loss:.4f}")
    
    # Retrain on the full outer training set and evaluate on the outer validation set
    model = OHEBase(input_size, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_function(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = loss_function(val_outputs, y_val).item()
        outer_fold_results.append(val_loss)
    print(f"Outer Fold Validation Loss: {val_loss:.4f}")

# Evaluate the final model on the evaluation set
model = OHEBase(input_size, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(train_encodings)
    loss = loss_function(outputs, train_targets)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    eval_outputs = model(eval_encodings)
    eval_loss = loss_function(eval_outputs, eval_targets).item()
    print(f"Evaluation Set Loss: {eval_loss:.4f}")

print("Nested cross-validation complete.")