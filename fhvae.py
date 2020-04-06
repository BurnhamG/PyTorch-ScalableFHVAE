import torch.nn as nn


class FHVAE(nn.Module):
    def __init__(
        self,
        input_size: int,
        z1_hus: list,
        z2_hus: list,
        z1_dim: int,
        z2_dim: int,
        x_hus: list,
    ):
        raise NotImplementedError()
