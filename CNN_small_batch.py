"""
    Writing a convolution neutal network training with epochs
    2 Layers NN => Make this into 3 layers nn
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

# modifying vision daa to run it through models

import matplotlib.pyplot as plt
import numpy as np

# Supervise learning
def main():
    features = torch.Tensor([[0, 0, 1, 1], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1]])

    labels = torch.Tensor([0, 1, 1, 0])

    # create a training set with features and labels
    train_set = TensorDataset(features, labels)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=False)

    # Build a model
    linear_layer1 = nn.Linear(
        4, 3
    )  # Linear layer that take in input of 4 and output of 2
    linear_layer2 = nn.Linear(
        3, 2
    )  # Linear layer that take in input of 2 and output of 1
    linear_layer3 = nn.Linear(2, 1)
    sigmoid = (
        nn.Sigmoid()
    )  # The output from layer 1 will go through the sigmoid function
    # to classify as 1 or 0

    # Training/Updates/Optimize parameters
    EPOCH_NUM = 5
    LR = 1e-1
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(linear_layer1.parameters(), lr=LR)

    for epoch in range(EPOCH_NUM):
        train_loader_iter = iter(train_loader)  # batch size of 4

        for batch_idx, (feature, label) in enumerate(train_loader_iter):
            linear_layer1.zero_grad()  # Reset layer1 for gradient calculation
            feature, label = Variable(feature.float()), Variable(
                label.float()
            )  # grab from small batch

        # Calculate the output with feedforwarding
        linear_layer1_output = linear_layer1(feature)
        sigmoid_layer1 = sigmoid(linear_layer1_output)
        linear_layer2_output = linear_layer2(sigmoid_layer1)
        sigmoid_layer2 = sigmoid(linear_layer2_output)
        linear_layer3_output = linear_layer3(sigmoid_layer2)
        sigmoid_layer3 = sigmoid(linear_layer3_output)

        # Calculate the loss function, gradient of loss and back propagation
        loss = loss_function(sigmoid_layer3, feature)
        loss.backward()
        optimizer.step()  # Update the weights and bias of layer 1?

        print("---------------------------------")
        print(
            "Output (UPDATE: Epoch #"
            + str(epoch + 1)
            + ", Batch #"
            + str(batch_idx + 1)
            + "):"
        )
        print(
            sigmoid(
                linear_layer3(
                    sigmoid(linear_layer2(sigmoid(linear_layer1(Variable(features)))))
                )
            )
        )  # Should get closer to the actual labels
    print("====================================")


if __name__ == "__main__":
    main()
