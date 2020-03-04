import numpy as np
import torch

from torch.nn import Linear, Conv2d, ConvTranspose2d
from torch.nn import init


class DenseLatent(torch.nn.Module):
    def __init__(
        self,
        inputs,
        num_outputs,
        weights_initializer=init.xavier_uniform_(),
        biases_initializer=init.zeros_(),
    ):
        super().__init__()
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer

    def forward(self, x):
        pass

    """A latent variable layer"""
    mu = Linear(inputs, num_outputs)
