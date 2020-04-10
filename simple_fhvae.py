import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from typing import List


class SimpleFHVAE(nn.Module):
    def __init__(
        self,
        input_size,
        z1_hus=[128, 128],
        z2_hus=[128, 128],
        z1_dim=16,
        z2_dim=16,
        x_hus=[128, 128],
    ):
        super().__init__()

        # priors
        self.pz1 = [0.0, np.log(1.0 ** 2).astype(np.float32)]
        self.pmu2 = [0.0, np.log(1.0 ** 2).astype(np.float32)]

        # encoder/decoder arch
        self.z1_hus = z1_hus
        self.z2_hus = z2_hus
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.x_hus = x_hus
        self.z1_pre_encoder = LatentSegPreEncoder(input_size, self.z1_hus)
        self.z2_pre_encoder = LatentSeqPreEncoder(input_size, self.z2_hus)
        self.z1_gauss_layer = GaussianLayer(input_size, self.z1_dim)
        self.z2_gauss_layer = GaussianLayer(input_size, self.z2_dim)
        self.pre_decoder = PreDecoder(input_size, self.x_hus)
        self.loss = nn.CrossEntropyLoss()

    def mu2_lookup(
        self, mu_idx: torch.Tensor, z2_dim: int, num_seqs: int, init_std: float = 1.0
    ):
        """Mu2 posterior mean lookup table

        Args:
            mu_idx:   Int tensor of shape (bs,). Index for mu2_table
            z2_dim:   Z2 dimension
            num_seqs: Lookup table size
            init_std: Standard deviation for lookup table initialization

        """
        mu2_table = torch.empty([num_seqs, z2_dim]).normal_(mean=4, std=init_std)
        mu2_table.requires_grad = True
        mu2 = torch.gather(mu2_table, mu_idx)
        return mu2_table, mu2

    def log_gauss(self, x, mu=0.0, logvar=0.0):
        """Compute log N(x; mu, exp(logvar))"""
        return -0.5 * (
            np.log(2 * np.pi) + logvar + torch.pow(x - mu, 2) / torch.exp(logvar)
        )

    def kld(self, p_mu, p_logvar, q_mu, q_logvar):
        """Compute D_KL(p || q) of two Gaussians"""
        return -0.5 * (
            1
            + p_logvar
            - q_logvar
            - (torch.pow(p_mu - q_mu, 2) + torch.exp(p_logvar)) / torch.exp(q_logvar)
        )

    def forward(
        self, x: torch.Tensor, mu_idx: torch.Tensor, num_seqs: int, num_segs: int
    ):
        """Forward pass through the network

        Args:
            x:        Input data
            mu_idx:   Int tensor of shape (bs,). Index for mu2_table
            num_seqs: Size of mu2 lookup table
            num_segs: Number of audio segments

        Returns:
            Variational lower bound and discriminative loss

        """
        mu2_table, mu2 = self.mu2_lookup(mu_idx, self.z2_dim, num_seqs)
        # z2 prior
        pz2 = [self.mu2, np.log(0.0 ** 2).astype(np.float32)]

        z2_pre_out = self.z2_pre_encoder(x)
        z2_mu, z2_logvar, z2_sample = self.gauss_layer(z2_pre_out, self.z2_dim)
        qz2_x = [z2_mu, z2_logvar]

        z1_pre_out = self.z1_pre_encoder(x, z2_sample)
        z1_mu, z1_logvar, z1_sample = self.gauss_layer(z1_pre_out, self.z1_dim)
        qz1_x = [z1_mu, z1_logvar]

        x_pre_out = self.pre_decoder(z1_sample, z2_sample, self.z1_hus)
        T, F = x.shape[1:]
        x_mu, x_logvar, x_sample = self.gauss_layer(x_pre_out, T * F)
        x_mu = torch.reshape(x_mu, (-1, T, F))
        x_logvar = torch.reshape(x_logvar, (-1, T, F))
        x_sample = torch.reshape(x_sample, (-1, T, F))
        px_z = [x_mu, x_logvar]

        # variational lower bound
        log_pmu2 = torch.sum(self.log_gauss(mu2, self.pmu2[0], self.pmu2[1]), dim=1)
        neg_kld_z2 = -1 * torch.sum(self.kld(qz2_x[0], qz2_x[1], pz2[0], pz2[1]), dim=1)
        neg_kld_z1 = -1 * torch.sum(
            self.kld(qz1_x[0], qz1_x[1], self.pz1[0], self.pz1[1]), dim=1
        )
        log_px_z = torch.sum(self.log_gauss(x, px_z[0], px_z[1]), dim=(1, 2))
        lower_bound = log_px_z + neg_kld_z1 + neg_kld_z2 + log_pmu2 / num_segs

        # discriminative loss
        logits = torch.unsqueeze(self.qz2_x[0], 1) - torch.unsqueeze(self.mu2_table, 0)
        logits = -1 * torch.pow(logits, 2) / (2 * torch.exp(pz2[1]))
        logits = torch.sum(logits, dim=-1)
        log_qy = self.loss(input=logits, target=mu_idx)

        return lower_bound, log_qy, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2


class VariableLinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class LatentSegPreEncoder(nn.Module):
    """Pre-stochastic layer encoder for z1 (latent segment variable)

    Args:
        input_size: Size of input to first layer
        x:          Tensor of shape (bs, T, F) (z1)
        lat_seq:    Latent sequence variable (z2)
        hus:        List of numbers of FC layer hidden units

    Returns:
        out: last FC layer output

    """

    def __init__(self, input_size: int, hus: List[int] = None):
        super().__init__()
        if hus is None:
            self.hus = [1024, 1024]
        else:
            self.hus = hus
        self.fc1 = VariableLinearLayer(input_size, self.hus[0])
        self.fc2 = VariableLinearLayer(self.hus[0], self.hus[1])

    def forward(self, x: torch.Tensor, lat_seq: torch.Tensor):
        x = torch.reshape(x, (-1,))
        out = torch.cat([x, lat_seq])
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class LatentSeqPreEncoder(nn.Module):
    """Pre-stochastic layer encoder for z2 (latent sequence variable)

    Args:
        input_size: Size of first layer input
        hus:        List of numbers of layer hidden units

    Returns:
        out: Concatenation of hidden states of all layers

    """

    def __init__(self, input_size, hus: List[int] = None):
        super().__init__()
        if hus is None:
            hus = [1024, 1024]
        self.fc1 = VariableLinearLayer(input_size, hus[0])
        self.fc2 = VariableLinearLayer(hus[0], hus[1])

    def forward(self, x):
        out = torch.reshape(x, (-1,))
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class GaussianLayer(nn.Module):
    """Gaussian layer

    Args:
        input_size:  Size of input to first layer
        dim:         Dimension of output latent variables
        input_layer: Input layer

    Returns:
        Average, log variance, and a sample from the gaussian

    """

    def __init__(self, input_size: int, dim: int):
        super().__init__()
        self.mulayer = nn.Linear(input_size, dim)
        self.logvar_layer = nn.Linear(input_size, dim)

    def forward(self, input_layer: torch.Tensor):
        mu = self.mulayer(input_layer)
        logvar = self.logvar_layer(input_layer)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu, logvar, mu + eps * std


class PreDecoder(nn.Module):
    """Pre-stochastic layer decoder

    Args:
        input_size: Size of input data
        hus:        List of hidden units per fully-connected layer
        lat_seg:    Latent segment Tensor (z1)
        lat_seq:    Latent sequence Tensor (z2)

    Returns:
        out: Concatenation of hidden states of all layers

    """

    def __init__(self, input_size: int, hus: List[int] = None):
        super().__init__()
        if hus is None:
            hus = [1024, 1024]
        self.fc1 = VariableLinearLayer(input_size, hus[0])
        self.fc2 = VariableLinearLayer(hus[0], hus[1])

    def forward(self, lat_seg: torch.Tensor, lat_seq: torch.Tensor):
        out = torch.cat([lat_seg, lat_seq])
        out = self.fc1(out)
        out = self.fc2(out)
        return out
