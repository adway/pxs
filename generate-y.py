import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

X = pd.read_csv('~/data/nhanes/full_data_sim.csv', index_col=False)
X = X.drop(['Unnamed: 0', 'RIDSTATR'], axis = 1)
X_stand = (X - X.mean()) / X.std()

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        # Initialize weights and biases with uniform random values
        nn.init.uniform_(self.hidden.weight, -0.5, 0.5)
        nn.init.uniform_(self.hidden.bias, -1, 1)
        nn.init.uniform_(self.output.weight, -1, 1)
        nn.init.uniform_(self.output.bias, -1, 1)

    def forward(self, x):
        hidden_output = F.relu(self.hidden(x))
        final_output = self.output(hidden_output)
        scaled_output = final_output
        return scaled_output

input_size = X_stand.shape[1]
hidden_size = 5
output_size = 1

model = FeedforwardNN(input_size, hidden_size, output_size)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
X_tensor = torch.tensor(X_stand.values, dtype=torch.float32)

y = model(X_tensor)
y_vec = y.view(-1).detach().numpy()
np.savetxt(os.path.expanduser('~/data/nhanes/y.csv'), y_vec, delimiter = '')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y_vec_squashed = sigmoid(y_vec)
np.savetxt(os.path.expanduser('~/data/nhanes/y-squashed.csv'), y_vec_squashed, delimiter = '')

plt.clf()
sns.histplot(y_vec)
plt.clf()
sns.histplot(y_vec_squashed)

# Code to save and load NN
with open(os.path.expanduser('~/data/nhanes/model.pkl'), 'wb') as file:
    pickle.dump(model, file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)