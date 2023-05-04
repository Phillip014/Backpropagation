# 反向传播算法入门教程

## 目录
1. 介绍
2. 反向传播算法详解
3. 使用原始Python实现三层神经网络
4. 使用PyTorch实现三层神经网络

## 1. 介绍

反向传播（Backpropagation）是神经网络中一种非常重要的优化算法。它通过计算损失函数相对于每个权重的梯度来调整神经网络的参数，从而减小损失值。

## 2. 反向传播算法详解

反向传播的关键在于链式法则，它允许我们将复合函数的导数分解为其各个组成部分。具体来说，反向传播算法包括以下步骤：

1. 正向传播，计算网络的输出和损失值。
2. 反向传播误差，计算损失函数关于每个权重的梯度。
3. 更新权重，使用梯度下降或其他优化算法来调整权重。

## 3. 使用原始Python实现三层神经网络

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.weights_input_hidden = np.random.randn(input_nodes, hidden_nodes)
        self.weights_hidden_output = np.random.randn(hidden_nodes, output_nodes)
        
    def forward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights_input_hidden))
        output = sigmoid(np.dot(self.hidden, self.weights_hidden_output))
        return output

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            error_output = y - output
            delta_output = error_output * sigmoid_derivative(output)
            
            error_hidden = np.dot(delta_output, self.weights_hidden_output.T)
            delta_hidden = error_hidden * sigmoid_derivative(self.hidden)
            
            self.weights_input_hidden += learning_rate * np.dot(X.T, delta_hidden)
            self.weights_hidden_output += learning_rate * np.dot(self.hidden.T, delta_output)
            
            if epoch % 1000 == 0:
                loss = mse_loss(y, self.forward(X))
                print(f'Epoch {epoch}, Loss: {loss}')
```

## 4. 使用PyTorch实现三层神经网络
```python
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
2(X))
        return output

def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

input_nodes = 2
hidden_nodes = 4
output_nodes = 1

neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
optimizer = optim.SGD(neural_network.parameters(), lr=0.1)

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

epochs = 10000

for epoch in range(epochs):
    output = neural_network(X)
    loss = mse_loss(y, output)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print("Training finished.")
```


