import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(NeuralNetwork, self).__init__()
        
        self.layer1 = nn.Linear(input_nodes, hidden_nodes)
        self.layer2 = nn.Linear(hidden_nodes, output_nodes)
        self.activation = nn.Sigmoid()

    def forward(self, X):
        hidden = self.activation(self.layer1(X))
        output = self.activation(self.layer
