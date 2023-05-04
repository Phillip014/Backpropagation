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
