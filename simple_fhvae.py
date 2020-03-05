import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class SimpleFHVAE(nn.Module):
    def __init__(self, xin, xout, y, n, nmu2):
        super().__init__()

        # encoder/decoder arch
        self.z1_hus, self.z1_dim = [128, 128], 16
        self.z2_hus, self.z2_dim = [128, 128], 16
        self.x_hus = [128, 128]

        # observed vars
        self.xin = xin
        self.xout = xout
        self.y = y
        self.n = n
        self.nmu2 = nmu2

        # latent vars
        (
            self.mu2_table,
            self.mu2,
            self.qz2_x,
            self.z2_sample,
            self.qzi_x,
            self.z1_sample,
            self.px_z,
            self.x_sample,
        ) = self.net(
            self.xin,
            self.xout,
            self.y,
            self.nmu2,
            self.z1_hus,
            self.z1_dim,
            self.z2_hus,
            self.z2_dim,
            self.x_hus,
        )
