"""
    Tutorials: https://lelon.io/blog/pytorch-baby-steps
"""
import torch # Tensor Package for GPU
from torch.autograd import Variable # For computational graphs
import torch.nn as nn # Neural network package
import torch.nn.functional as F # Non-linear package
import torch.optim as optim # Optimiazation Package
from torch.utils.data import Dataset, TensorDataset, DataLoader # Data Processing package
import torchvision # data Vision package
import torchvision.transforms as transforms # modifying vision daa to run it through models

import matplotlib.pyplot as plt
import numpy as np

def main():
    # # 1: Create a tensor a nd put in a Variable for training later
    # x1 = torch.Tensor([1,2,3,4])
    # x1_node = Variable(x1, requires_grad=True)
    # print(x1_node)
    # y1_node = x1_node
    # print(y1_node)

    # # 2: Create a Linear Regression Model
    # x1 = torch.Tensor([1,2,3,4,5]) # Create a tensor
    # x1_var = Variable(x1, requires_grad=True) # Autograd allow freezing the layer
    #
    # # Crate a linear regression layer w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + b => Take in 5 input and provide 1 output prediction
    # linear_reg_layer1 = nn.Linear(5,1) # The weight is set to randome
    #
    # predicted_y = linear_reg_layer1(x1_var)
    # print(f"The prediction of linear regression is {predicted_y}")

    # # 3 Calculate graident of linear regression layer with x1 input
    # x1 = torch.Tensor([1,2,3,4,5])
    # x1_var = Variable(x1, requires_grad=True) # Make a tensor into a variable for trainable purposes, we allow the input to calculate the gradient
    #
    # linear_layer1 = nn.Linear(5,1) # Create a Linear model, the weight at this point is random
    # labels = Variable(torch.Tensor([0]), requires_grad=False)
    # predicted_result = linear_layer1(x1_var)
    #
    # # # Calcualate the gradient of the linear layer with respect to the input layer.
    # # print(f"Since we have not calculate the gradient of x1_var yet, now it will be {x1_var.grad}")
    # #
    # # # Using backprop, we can calculate the gradient with respect to the output
    # # predicted_result.backward()
    # # print(f"The gradient of the layer with respect to the input is: {x1_var.grad}") # Again, the gradient is various since the weight is various
    #
    # # 4: Calculating the LOSS Function
    # loss_function = nn.MSELoss() # Create a mean square loss function, take in predicted_result and labels, return the loss
    # loss = loss_function(predicted_result, labels)
    # print(f"The loss results is {loss}") # calculate the mean square of all 5 param
    #
    # # 5: Update the weight with SGD.
    # optimizer = optim.SGD(linear_layer1.parameters(), lr=1e-1) # type, what layer, learning rate
    # # create the optimizer object for updateing weight, bias with SGD, update the weights only in the linear layer1 with learning rate of 10^-9. Nothing is learn yet
    #
    # loss.backward() # Calculate the gradient of the loss first
    #
    # # Before updating
    # print(f"The weights and bias before update are: {linear_layer1.weight} and {linear_layer1.bias}")
    # print(f"The output is: {linear_layer1(x1_var)}")
    #
    # optimizer.step()
    #
    # print("\n")
    # print(f"The weights and bias before update are: {linear_layer1.weight} and {linear_layer1.bias}")
    # print(f"The output is: {linear_layer1(x1_var)}")

    ############################## Fully Updates with Stochastic Gradient Descent with 6 epochs
    x1 = torch.Tensor([1,2,3,4,56])
    x1_var = Variable(x1, requires_grad=False)

    linear_layer1 = nn.Linear(5,1) # Create a layer model
    labels = Variable(torch.Tensor([0]), requires_grad=False) # Generate output

    # Using feedforward to calculate the output
    print(f"Before Training, output are: {linear_layer1(x1_var)}")
    print()

    EPOCH_NUM = 6
    LR = 1e-4
    loss_function = nn.MSELoss() # Create a loss function
    optimizer = optim.SGD(linear_layer1.parameters(), lr=LR) # Create an optimizer

    for epoch in range(EPOCH_NUM):
        linear_layer1.zero_grad() # Refresh the gradient everytime
        predictions = linear_layer1(x1_var)
        loss = loss_function(predictions, labels)
        loss.backward() # Calculate the gradient
        optimizer.step()

        print(f"Epoch {epoch} with output: {linear_layer1(x1_var)}") # Note that the output gets better

if __name__ == "__main__":
    main()