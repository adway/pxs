import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

X = pd.read_csv('~/data/nhanes/full_data_sim.csv', index_col=False)
X = X.drop(['Unnamed: 0', 'RIDSTATR'], axis = 1)
X_stand = (X - X.mean()) / X.std()

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        # Initialize weights and biases with uniform random values
        nn.init.uniform_(self.hidden.weight, -0.1, 0.1)
        nn.init.uniform_(self.hidden.bias, -0.1, 0.1)
        nn.init.uniform_(self.output.weight, -0.1, 0.1)
        nn.init.uniform_(self.output.bias, -0.1, 0.1)

    def forward(self, x):
        hidden_output = F.relu(self.hidden(x))
        final_output = self.output(hidden_output)
        scaled_output = final_output*0.1
        return final_output

input_size = X_stand.shape[1]
hidden_size = 5
output_size = 1

model = FeedforwardNN(input_size, hidden_size, output_size)
X_tensor = torch.tensor(X_stand.values, dtype=torch.float32)

y = model(X_tensor)
y_vec = y.view(-1).detach().numpy()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y_vec_squashed = sigmoid(y_vec)



print(y_vec)
plt.clf()
sns.histplot(y_vec_squashed)

