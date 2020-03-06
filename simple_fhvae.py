import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

import costs

loss = nn.CrossEntropyLoss()


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

        # priors
        self.pz1 = [0.0, np.log(1.0 ** 2).astype(np.float32)]
        self.pz2 = [self.mu2, np.log(0.0 ** 2).astype(np.float32)]
        self.pmu2 = [0.0, np.log(1.0 ** 2).astype(np.float32)]

        # variational lower bound
        self.log_pmu2 = torch.sum(
            log_gauss(self.mu2, self.pmu2[0], self.pmu2[1]), dim=1
        )
        self.neg_kld_z2 = -1 * torch.sum(
            kld(self.qz2_x[0], self.qz2_x[1], self.pz2[0], self.pz2[1]), dim=1
        )
        self.neg_kld_z1 = -1 * torch.sum(
            kld(self.qz1_x[0], self.qz1_x[1], self.pz1[0], self.pz1[1]), dim=1
        )
        self.log_px_z = torch.sum(
            log_gauss(xout, self.px_z[0], self.px_z[1]), dim=(1, 2)
        )
        self.lb = self.log_px_z + self.neg_kld_z1 + self.neg_kld_z2 + self.log_pmu2 / n

        # discriminative loss
        logits = torch.unsqueeze(self.qz2_x[0], 1) - torch.unsqueeze(self.mu2_table, 0)
        logits = -1 * torch.pow(logits, 2) / (2 * torch.exp(self.pz2[1]))
        logits = torch.sum(logits, dim=-1)
        self.log_qy = loss(input=logits, target=y)

    def net(self, xin, xout, y, nmu2, z1_hus, z1_dim, z2_hus, z2_dim, x_hus):
        """Initialize the network"""
        self.mu2_table, self.mu2 = mu2_lookup(self.y, z2_dim, self.nmu2)

        z2_pre_out = z2_pre_encoder(self.xin, z2_hus)
        z2_mu, z2_logvar, z2_sample = gauss_layer(z2_preout, z2_dim)
        qz2_x = [z1_mu, z1_logvar]

        z1_pre_out = z1_pre_encoder(xin, z2_sample, z1_hus)
        z1_mu, z1_logvar, z1_sample = gauss_layer(z1_pre_out, z1_dim)
        qz1_x = [z1_mu, z1_logvar]

        x_pre_out = pre_decoder(z1_sample, z2_sample, x_hus)
        T, F = list(xout.size())[1:]
        x_mu, x_logvar, x_sample = gauss_layer(x_pre_out, T * F)
        x_mu = torch.reshape(x_mu, (-1, T, F))
        x_logvar = torch.reshape(x_logvar, (-1, T, F))
        x_sample = torch.reshape(x_sample, (-1, T, F))
        # p(x|z_1, z_2)
        px_z = [x_mu, x_logvar]

        return mu2_table, mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample

    def __str__(self):
        msg = ""
        msg += "\nFactorized Hierarchical Variational Autoencoder:"
        msg += "\n  Priors (mean/logvar):"
        msg += "\n    pz1: %s" % str(self.pz1)
        msg += "\n    pz2: %s" % str(self.pz2)
        msg += "\n    pmu2: %s" % str(self.pmu2)
        msg += "\n  Observed Variables:"
        msg += "\n    xin: %s" % self.xin
        msg += "\n    xout: %s" % self.xout
        msg += "\n    y: %s" % self.y
        msg += "\n    n: %s" % self.n
        msg += "\n  Encoder/Decoder Architectures:"
        msg += "\n    z1 encoder:"
        msg += "\n      FC hidden units: %s" % str(self.z1_hus)
        msg += "\n      latent dim: %s" % self.z1_dim
        msg += "\n    z2 encoder:"
        msg += "\n      FC hidden units: %s" % str(self.z2_hus)
        msg += "\n      latent dim: %s" % self.z2_dim
        msg += "\n    mu2 table size: %s" % self.nmu2
        msg += "\n    x decoder:"
        msg += "\n      FC hidden units: %s" % str(self.x_hus)
        msg += "\n  Outputs:"
        msg += "\n    qz1_x: %s" % str(self.qz1_x)
        msg += "\n    qz2_x: %s" % str(self.qz2_x)
        msg += "\n    mu2: %s" % str(self.mu2)
        msg += "\n    px_z: %s" % str(self.px_z)
        msg += "\n    z1_sample: %s" % str(self.z1_sample)
        msg += "\n    z2_sample: %s" % str(self.z2_sample)
        msg += "\n    x_sample: %s" % str(self.x_sample)
        msg += "\n  Losses:"
        msg += "\n    lb: %s" % str(self.lb)
        msg += "\n    log_px_z: %s" % str(self.log_px_z)
        msg += "\n    neg_kld_z1: %s" % str(self.neg_kld_z1)
        msg += "\n    neg_kld_z2: %s" % str(self.neg_kld_z2)
        msg += "\n    log_pmu2: %s" % str(self.log_pmu2)
        msg += "\n    log_qy: %s" % str(self.log_qy)
        msg += "\n  Parameters:"
        for name, param in self.state_dict().items():
            msg += "\n    %s, %s" % (param.name, param.size())
        return msg

    def forward(self, x):
        pass


def mu2_lookup(y, z2_dim, nmu2, init_std=1.0):
    """
    Mu2 posterior mean lookup table
    Args:
        y(torch.Tensor): int tensor of shape (bs,). Index for mu2_table
        z2_dim(int): z2 dimension
        nmu2(int): lookup table size
    """
    mu2_table = torch.empty([nmu2, z2_dim], requires_grad=True).normal_(
        mean=4, std=init_std
    )
    mu2 = torch.gather(mu2_table, y)
    return mu2_table, mu2


def z1_pre_encoder(x, z2, hus=[1024, 1024]):
    """
    Pre-stochastic layer encoder for z1 (latent segment variable)
    Args:
        x(torch.Tensor): tensor of shape (bs, T, F)
        z2(torch.Tensor): tensor of shape (bs, D1)
        hus(list): list of numbers of FC layer hidden units
    Return:
        out(torch.Tensor): last FC layer output
    """
    T, F = list(x.size())[1:]
    x = torch.reshape(x, (-1, T * F))
    out = torch.cat([x, z2], dim=-1)
    for i, hu in enumerate(hus):
        out = nn.Linear(np.prod(out.shape[1:]), hu)(out)
        out = F.relu(out)
    return out


def z2_pre_encoder(x, hus=[1024, 1024]):
    """
    Pre-stochastic layer encoder for z2 (latent sequence variable)
    Args:
        x(torch.Tensor): tensor of shape (bs, T,F)
        hus(list): list of numbers of LSTM layer hidden units
    Return:
        out(torch.Tensor): concatenation of hidden states of all LSTM layers
    """
    T, F = list(x.size())[1:]
    out = torch.reshape(x, (-1, T * F))
    for i, hu in enumerate(hus):
        out = nn.Linear(np.prod(out.shape[1:]), hu)(out)
        out = F.relu(out)
    return out


def gauss_layer(inp, dim):
    """
    Gaussian layer
    Args:
        inp(torch.Tensor): input to Gaussian layer
        dim(int): dimension of output latent variables
    """
    mu = nn.Linear(np.prod(inp.shape[1:]), dim)(inp)
    logvar = nn.Linear(np.prod(inp.shape[1:]), dim)(inp)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu, logvar, mu + eps * std


def pre_decoder(z1, z2, hus=[1024, 1024]):
    """
    Pre-stochastic layer decoder
    Args:
        z1(torch.Tensor)
        z2(torch.Tensor)
        x(torch.Tensor): tensor of shape (bs, T, F). Only shape is used
        hus(list)
    """
    out = torch.cat([z1, z2], dim=-1)
    for i, hu in enumerate(hus):
        out = nn.Linear(np.prod(out.shape[1:]), hu)(out)
        out = F.relu(out)
    return out


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
