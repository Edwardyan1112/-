import numpy as np


class Layer:
    def __init__(self):
        pass
    
    def forward(self, input):
        return input
    
    def backward(self, input, grad_output):
        pass
    
    
class ReLU(Layer):
    def __init__(self):
        pass
    
    def forward(self, input):
        return np.maximum(input, 0)
    
    
    def backward(self, input, grad_output):
        return (input > 0) * grad_output
    
    
class Sigmoid(Layer):
    def __init__(self):
        pass
    
    def _sigmoid(self, input):
        return 1.0 / (1 + np.exp(-input))
        
    
    def forward(self, input):
        return self._sigmoid(input)
    
    def backward(self, input, grad_output):
        sigmoid_output = self._sigmoid(input)
        return sigmoid_output * (1 - sigmoid_output) * grad_output
    
    
class Tanh(Layer):
    def __init__(self):
        pass
    
    def _tanh(self, x):
        # e_x = np.exp(x)
        # e_minus_x = np.exp(-x)
        
        # return (e_x - e_minus_x) / (e_x + e_minus_x)
        return np.tanh(x)
    
    def forward(self, input):
        return self._tanh(input)
    
    def backward(self, input, grad_output):
        return (1 - self._tanh(input) ** 2) * grad_output
    
    
class Dense(Layer):
    def __init__(self, dim_in, dim_out, lr):
        self.lr = lr
        self.weight = np.random.randn(dim_in, dim_out)
        self.bias = np.zeros(dim_out)
        
    def forward(self, input):
        return np.dot(input, self.weight) + self.bias
    
    def backward(self, input, grad_output):
        grad_in = np.dot(grad_output, self.weight.T)
        
        grad_weights = np.dot(input.T, grad_output) / input.shape[0]
        grad_bias = grad_output.mean(axis=0)
        
        self.weight = self.weight - self.lr * grad_weights
        self.bias = self.bias - self.lr * grad_bias
        
        return grad_in
    
    
def forward(networks, X):
    activations = []
    
    input = X
    for network in networks:
        activations.append(network.forward(input))
        input = activations[-1]
        
    return activations


def predict(networks, X):
    return forward(networks, X)[-1]


def train(networks, X, y):
    activations = forward(networks, X)
    activations = [X] + activations
    
    logits = activations[-1]
    loss = np.square(logits - y).sum()
    loss_grad = 2.0 * np.abs(logits - y)
    
    for layer_idx in range(len(networks))[::-1]:
        loss_grad = networks[layer_idx].backward(activations[layer_idx], loss_grad)
    
    return loss

networks = []
networks.append(Dense(64, 32, 0.01))
networks.append(Tanh())
networks.append(Dense(32, 1, 0.01))
networks.append(Sigmoid())


class Linear():
    def __init__(self, dim_in, dim_out, lr):
        self.lr = lr
        
        self.weights = np.random.randn(dim_in, dim_out)
        self.bias = np.zeros(dim_out)
        
    def forward(self, model_input):
        return np.dot(model_input, self.weights) + self.bias
    

    def backward(self, model_input, grad_out):
        grad_in = np.dot(grad_out, self.weights.T)
        
        grad_weight = np.dot(model_input.T, grad_out) / model_input.shape[0]
        grad_bias = np.mean(grad_out, axis=0)
        
        self.weights = self.weights - self.lr * grad_weight
        self.bias = self.bias - self.lr * grad_bias
        
        return grad_in