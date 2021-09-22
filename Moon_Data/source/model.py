import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):

    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Defines layers of a neural network.
        :param input_dim: Number of input features
        :param hidden_dim: Size of hidden layer(s)
        :param output_dim: Number of outputs
        """
        super(SimpleNet, self).__init__()

        # define all layers, here
        self.l_0 = nn.Linear(input_dim, hidden_dim)
        self.l_1 = nn.Linear(hidden_dim, output_dim)
        self.a_2 = nn.Sigmoid()

    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """Feedforward behavior of the net.
        :param x: A batch of input features
        :return: A single, sigmoid activated value
        """
        # your code, here
        z_0 = F.relu(self.l_0(x))
        z_1 = F.relu(self.l_1(z_0))
        return self.a_2(z_1)
