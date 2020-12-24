"""
    Create a 4 layers neural networks with class
    TODO: Cifar
          MNIST dataset with convolution neural network
"""
import torch  # Tensor Package for GPU
from torch.autograd import Variable  # For computational graphs
import torch.nn as nn  # Neural network package
import torch.nn.functional as F  # Non-linear package
import torch.optim as optim  # Optimiazation Package
from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
)  # Data Processing package
import torchvision  # data Vision package
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

#################################### Create training data
features = torch.Tensor([[0, 0, 1, 1], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
labels = torch.Tensor([0, 1, 1, 0]) #There are only 4 labels so there should be only 4 pairs

# create a training set with features and labels
train_set = TensorDataset(features, labels) # Paring the features and labels
train_loader = DataLoader(train_set, batch_size = 4, shuffle=False)
# Number of batch size = number of elements in 1 training batch is 4


# Create a neural network with class
class Net(nn.Module): #standard?
    def __init__(self): # Generate layers objects
        super(Net, self).__init__()  # Super class inheritant the initial and can add in more feature? Override the module
        self.fc1 = nn.Linear(4, 3)  # layer take input of dim 4x1 and output 3x1 dim
        self.fc2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def feedforward(self, x):  # Feedforward to calculate the output
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def main():
    print("Start Training")
    net = Net()

    # Initialize Hyper Parameter, epoch, learning rate, loss function, optimizer
    EPOCH_NUM = 10 # run 3 time
    LR = 1e-1
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr = LR) # Use for updating the weights and bias of all layers :)

    # Training with stochastic gradient descent
    for epoch in range(EPOCH_NUM):
        train_loader_iter = iter(train_loader) # make training pairs into iteration to get batch index
        # enumeration return the index and the value itself
        for batch_idx, (feature, label) in enumerate(train_loader_iter):
            # zero the gradient
            net.zero_grad() # zero the gradient of all weights and bias

            # make the features and labels into Variable to calculate the gradient
            feature, label = Variable(feature), Variable(label)

            # Feed forward to calculate the output
            output = net.feedforward(feature)
            print(f"Testing 1 {output}")

            # Loss
            loss = loss_function(output, label)

            # gradient of the loss with backprop
            loss.backward()

            # step the optimizer to update the weights and bias in the network in all 3 layers
            optimizer.step()

            print("-------------------------")
            print("---------------------------------")
            print(
                "Output (UPDATE: Epoch #"
                + str(epoch + 1)
                + ", Batch #"
                + str(batch_idx + 1)
                + "):"
            )
            print(net.feedforward(feature))  # Since the weights and biases in network has been updated, this value will different from 1
    print("====================================")

if __name__ == "__main__":
    main()
