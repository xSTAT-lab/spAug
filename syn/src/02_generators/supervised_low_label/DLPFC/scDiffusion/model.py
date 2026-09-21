"""Project-local scDiffusion neural network components.

Provides time embeddings, the Cell_Unet backbone, classifier guidance, and
the VAE components used by the task entry points.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ============================================================
# helper functions
# ============================================================

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    positive timestepencode (Vaswani et al.  ).

    Args:
        timesteps: (B,) 1-D timestep index  (canas number)
        dim:       encodedimension
        max_period:  

    Returns:
        (B, dim)  encode 
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """fornon- batch dimension mean."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# ============================================================
#  
# ============================================================

class TimeEmbedding(nn.Module):
    """
    timestep :
    sinusoidal encoding -> Linear -> SiLU -> Linear

    input: (B, 1) timestep index
    output: (B, hidden_dim) timestep 
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, 1) -> squeeze -> (B,) -> timestep_embedding -> (B, dim)
        emb = timestep_embedding(t.squeeze(-1), self.hidden_dim)
        return self.time_embed(emb)


class ResidualBlock(nn.Module):
    """
    timestepcondition :

    h = Linear(x) + Project(SiLU(time_emb))
    h = LayerNorm(h)
    h = SiLU(h)
    h = Dropout(h)

    Args:
        in_features:  inputdimension
        out_features: output dimension
        time_features: timestep dimension (hidden_num[0])
    """

    def __init__(self, in_features: int, out_features: int, time_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_features, out_features),
        )
        self.act = nn.SiLU()
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        h = h + self.emb_layer(emb)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return h


class Cell_Unet(nn.Module):
    """
    scDiffusion denoising network: encoder-decoder + Skip Connections.

     :
      encoder: input -> [ResBlock] x len(hidden_num) -> bottleneck
      decoder: bottleneck -> [ResBlock + skip] x len(hidden_num)-1 -> output
      output : Linear -> LayerNorm -> SiLU -> Linear(n_genes)

    Args:
        input_dim: gene count (input/output dimension)
        hidden_num:  dimensionlist, such as [512, 512, 256, 128]
        dropout:   dropout   ( parameter, default 0)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_num: list = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_num is None:
            hidden_num = [512, 512, 256, 128]
        self.hidden_num = hidden_num
        self.input_dim = input_dim

        # timestep  (dimension = hidden_num[0])
        self.time_embedding = TimeEmbedding(hidden_num[0])

        # ---- encoder ----
        self.layers = nn.ModuleList()
        #  : input_dim -> hidden_num[0]
        self.layers.append(ResidualBlock(input_dim, hidden_num[0], hidden_num[0]))
        # after : hidden_num[i] -> hidden_num[i+1]
        for i in range(len(hidden_num) - 1):
            self.layers.append(
                ResidualBlock(hidden_num[i], hidden_num[i + 1], hidden_num[0])
            )

        # ---- decoder (contains skip connection) ----
        self.reverse_layers = nn.ModuleList()
        for i in reversed(range(len(hidden_num) - 1)):
            self.reverse_layers.append(
                ResidualBlock(hidden_num[i + 1], hidden_num[i], hidden_num[0])
            )

        # ---- output  ----
        self.out1 = nn.Linear(hidden_num[0], int(hidden_num[1] * 2))
        self.norm_out = nn.LayerNorm(int(hidden_num[1] * 2))
        self.out2 = nn.Linear(int(hidden_num[1] * 2), input_dim, bias=True)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x_input: torch.Tensor, t: torch.Tensor, y=None) -> torch.Tensor:
        """
        beforeto : predicted noise epsilon.

        Args:
            x_input: (B, n_genes)  input
            t:       (B, 1)       timestep

        Returns:
            (B, n_genes) predictofnoise
        """
        emb = self.time_embedding(t)
        x = x_input.float()

        # encoder:  sample, savein status
        history = []
        for layer in self.layers:
            x = layer(x, emb)
            history.append(x)

        #  after requires skip (bottleneck)
        history.pop()

        # decoder:  sample + skip connection
        for layer in self.reverse_layers:
            x = layer(x, emb)
            x = x + history.pop()  # skip

        # output 
        x = self.out1(x)
        x = self.norm_out(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.out2(x)
        return x


class DiffusionClassifier(nn.Module):
    """
    condition classification : classify noisy expression/latent x_t at timestep t.

    Published-design-aligned port of `guided_diffusion.cell_model.Cell_classifier`.
    scDiffusion ofconditiongenerate  classifier guidance: in to use
    gradient_x log p(condition | x_t, t)  positivesample to.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_num: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_num is None:
            hidden_num = [512, 512, 256, 128]
        if len(hidden_num) < 3:
            raise ValueError("DiffusionClassifier requires at least three hidden dimensions")
        self.input_dim = int(input_dim)
        self.n_classes = int(n_classes)
        self.hidden_num = list(hidden_num)
        self.drop_rate = float(dropout)

        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_num[0], self.hidden_num[0]),
            nn.SiLU(),
            nn.Linear(self.hidden_num[0], self.hidden_num[0]),
        )

        self.fc1 = nn.Linear(self.input_dim, self.hidden_num[0], bias=True)
        self.emb_layers1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_num[0], self.hidden_num[0]),
        )
        self.norm1 = nn.BatchNorm1d(self.hidden_num[0])

        self.fc2 = nn.Linear(self.hidden_num[0], self.hidden_num[1], bias=True)
        self.emb_layers2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_num[0], self.hidden_num[1]),
        )
        self.norm2 = nn.BatchNorm1d(self.hidden_num[1])

        self.fc3 = nn.Linear(self.hidden_num[1], self.hidden_num[2], bias=True)
        self.emb_layers3 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_num[0], self.hidden_num[2]),
        )
        self.norm3 = nn.BatchNorm1d(self.hidden_num[2])

        self.act = nn.SiLU()
        self.drop = nn.Dropout(self.drop_rate)
        self.out = nn.Linear(self.hidden_num[2], self.n_classes, bias=True)

    def forward(self, x_input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(timestep_embedding(t.squeeze(-1), self.hidden_num[0]))
        x = x_input.float()

        x = self.fc1(x)
        x = x + self.emb_layers1(emb)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = x + self.emb_layers2(emb)
        x = self.norm2(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc3(x)
        # The published Cell_classifier defines emb_layers3 and leaves it unused in forward.
        x = self.norm3(x)
        x = self.act(x)
        x = self.drop(x)
        return self.out(x)


# ============================================================
# VAE (Autoencoder  process)
# ============================================================

class Encoder(nn.Module):
    """
    VAE encoder (deterministic,   scimilarity Encoder).

     : input_dropout -> [Linear -> BN -> PReLU] x n_layers -> Linear(latent_dim) -> L2 normalize

    Args:
        n_genes:    input gene count
        latent_dim: latent-space dimension (default 128)
        hidden_dim: hidden layer dimensionlist (default [1024, 1024, 1024])
        dropout:    hidden layer dropout  
        input_dropout: input  dropout  
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: Optional[List[int]] = None,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [1024, 1024, 1024]
        self.latent_dim = latent_dim

        layers = []
        in_dim = n_genes
        for i, h_dim in enumerate(hidden_dim):
            if i == 0:
                layers.append(nn.Dropout(p=input_dropout))
            else:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.PReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.network(x)
        return F.normalize(z, p=2, dim=1)  # L2 normalization (follows the published design)


class Decoder(nn.Module):
    """
    VAE decoder.

     : [Linear -> BN -> PReLU] x n_layers -> Linear(n_genes)

    Args:
        n_genes:    output gene count
        latent_dim: latent-space dimension
        hidden_dim: hidden layer dimensionlist (reverse of encoder)
        dropout:    dropout  
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [1024, 1024, 1024]

        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_genes))

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class GeneVAE(nn.Module):
    """
    gene expression VAE (deterministic encoder + L2 normalization ).

    The scDiffusion workflow uses this model as follows:
      1. training:  loss (MSE)
      2. encode: will gene expression to latent_dim  
      3.  model training and generation in latent space
      4.  : will latent  as gene expression

    Args:
        n_genes:    gene count
        latent_dim: latent-space dimension (default 128)
        hidden_dim: hidden layer dimension (default [1024, 1024, 1024])
        dropout:    dropout  
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [1024, 1024, 1024]
        self.n_genes = n_genes
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            input_dropout=dropout,
        )
        self.decoder = Decoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=list(reversed(hidden_dim)),
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """encode: x -> z (L2 normalization)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """ : z -> x_hat"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ beforeto: x -> z -> x_hat"""
        z = self.encode(x)
        return self.decode(z)

    def train_step(self, x: torch.Tensor, optimizer) -> float:
        """single diffusion training step"""
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
