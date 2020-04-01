import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class SimpleFHVAE(nn.Module):
    def __init__(
        self,
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
        self.z1_pre_encoder = LatentSegPreEncoder(self.z1_hus)
        self.z2_pre_encoder = LatentSeqPreEncoder(self.z2_hus)
        self.gauss_layer = GaussianLayer()
        self.pre_decoder = PreDecoder(self.x_hus)

    def mu2_lookup(mu_idx, z2_dim, num_seqs, init_std=1.0):
        """
        Mu2 posterior mean lookup table
        Args:
            mu_idx(torch.Tensor): int tensor of shape (bs,). Index for mu2_table
            z2_dim(int): z2 dimension
            num_seqs(int): lookup table size
        """
        mu2_table = torch.empty([num_seqs, z2_dim], requires_grad=True).normal_(
            mean=4, std=init_std
        )
        mu2 = torch.gather(mu2_table, mu_idx)
        return mu2_table, mu2

    def log_gauss(x, mu=0.0, logvar=0.0):
        """
        Compute log N(x; mu, exp(logvar))
        """
        return -0.5 * (
            np.log(2 * np.pi) + logvar + torch.pow(x - mu, 2) / torch.exp(logvar)
        )

    def kld(p_mu, p_logvar, q_mu, q_logvar):
        """
        Compute D_KL(p || q) of two Gaussians
        """
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
        loss = nn.CrossEntropyLoss()
        logits = torch.unsqueeze(self.qz2_x[0], 1) - torch.unsqueeze(self.mu2_table, 0)
        logits = -1 * torch.pow(logits, 2) / (2 * torch.exp(pz2[1]))
        logits = torch.sum(logits, dim=-1)
        log_qy = loss(input=logits, target=mu_idx)

        return lower_bound, log_qy, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2


class LatentSegPreEncoder(nn.Module):
    """
    Pre-stochastic layer encoder for z1 (latent segment variable)
    Args:
        x(torch.Tensor): tensor of shape (bs, T, F)
        lat_seq(torch.Tensor): latent sequence variable (z2)
        hus(list): list of numbers of FC layer hidden units
    Return:
        out(torch.Tensor): last FC layer output
    """

    def __init__(self, hus=[1024, 1024]):
        super().__init__()
        self.hus = hus

    def forward(self, x, lat_seq):
        x = torch.reshape(x, (-1,))
        out = torch.cat([x, lat_seq])
        for hu in self.hus:
            out = nn.Linear(np.prod(out.shape[1:]), hu)(out)
            out = F.relu(out)
        return out


class LatentSeqPreEncoder(nn.Module):
    """
    Pre-stochastic layer encoder for z2 (latent sequence variable)
    Args:
        hus(list): list of numbers of LSTM layer hidden units
    Return:
        out(torch.Tensor): concatenation of hidden states of all LSTM layers
    """

    def __init__(self, hus=[1024, 1024]):
        super().__init__()
        self.hus = hus

    def forward(self, x):
        out = torch.reshape(x, (-1,))
        for hu in self.hus:
            out = nn.Linear(np.prod(out.shape[1:]), hu)(out)
            out = F.relu(out)
        return out


class GaussianLayer(nn.Module):
    """
    Gaussian layer
    Args:
        input_layer(torch.Tensor): input layer
        dim(int): dimension of output latent variables
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_layer, dim):
        mu = nn.Linear(np.prod(input_layer.shape[1:]), dim)(input_layer)
        logvar = nn.Linear(np.prod(input_layer.shape[1:]), dim)(input_layer)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu, logvar, mu + eps * std


class PreDecoder(nn.Module):
    """
    Pre-stochastic layer decoder
    Args:
        lat_seg(torch.Tensor): latent segment Tensor (z1)
        lat_seq(torch.Tensor): latent sequence Tensor (z2)
        hus(list): list of hidden units per fully-connected layer
    """

    def __init__(self, hus=[1024, 1024]):
        super().__init__()
        self.hus = hus

    def forward(self, lat_seg, lat_seq):
        out = torch.cat([lat_seg, lat_seq])
        for hu in self.hus:
            out = nn.Linear(np.prod(out.shape[1:]), hu)(out)
            out = F.relu(out)
        return out
