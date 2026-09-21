"""Project-local scGAN and cscGAN-style WGAN-GP components.

The module provides unconditional and conditional generators, critics, and
training utilities following the published scGAN design.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Generator
# ============================================================

class ConditionalBatchNorm1d(nn.Module):
    """
    Conditional batch normalization used by the public cscGAN TensorFlow source.

    The original implementation keeps moving mean/variance per hidden layer and
    selects condition-specific offset/scale vectors by label. This module mirrors
    that mechanism for fully connected PyTorch activations.
    """

    def __init__(self, num_features: int, n_conditions: int, momentum: float = 0.001, eps: float = 1e-5):
        super().__init__()
        if n_conditions < 1:
            raise ValueError("ConditionalBatchNorm1d requires n_conditions >= 1")
        self.num_features = int(num_features)
        self.n_conditions = int(n_conditions)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.offset = nn.Embedding(self.n_conditions, self.num_features)
        self.scale = nn.Embedding(self.n_conditions, self.num_features)
        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_var", torch.ones(self.num_features))
        nn.init.zeros_(self.offset.weight)
        nn.init.ones_(self.scale.weight)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition is None:
            raise ValueError("ConditionalBatchNorm1d.forward requires condition")
        condition = condition.long().view(-1).to(x.device)
        if self.training and x.size(0) > 1:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean.detach())
            self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * var.detach())
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean.view(1, -1)) / torch.sqrt(var.view(1, -1) + self.eps)
        return x_norm * self.scale(condition) + self.offset(condition)


class Generator(nn.Module):
    """
    Unconditional generator: maps noise to gene expression.

     : latent_dim -> gen_hidden[0] -> ... -> gen_hidden[-1] -> n_genes
    hidden layer in each block: Linear -> BatchNorm1d -> ReLU
    output layer:     Linear -> output_activation (ReLU / Identity)
    LSN:        Library Size Normalization (optional, follows the published design)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        gen_hidden: list = None,
        n_genes: int = 3000,
        use_batch_norm: bool = True,
        output_activation: str = "relu",
        lsn_lib_size: float = None,
    ):
        """
        Args:
            latent_dim:          noise dimension
            gen_hidden:          units in each hidden layer,default [256, 512]
            n_genes:             output gene count
            use_batch_norm:      enable BatchNorm in hidden layers
            output_activation:   output activation "relu" / "tanh" / "none"
            lsn_lib_size:        LSN target library size; supply a positive value to enable LSN
        """
        super().__init__()
        if gen_hidden is None:
            gen_hidden = [256, 512]
        self.latent_dim = latent_dim
        self.n_genes = n_genes
        self.gen_hidden = gen_hidden
        self._output_activation = output_activation
        self.lsn_lib_size = lsn_lib_size

        layers = []
        in_dim = latent_dim
        for i, h_dim in enumerate(gen_hidden):
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim

        # output layer
        layers.append(nn.Linear(in_dim, n_genes))
        self.main = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialize (follows the published design)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def _apply_lsn(x: torch.Tensor, target_lib_size: torch.Tensor) -> torch.Tensor:
        """
        Library Size Normalization:  each ofoutput its and = target_lib_size.

        follows the published design:
            sigmas = reduce_sum(fake_outputs, axis=1)
            scale_ls = gamma / (sigmas + epsilon)
            fake_outputs = transpose(transpose(fake_outputs) * scale_ls)

        Args:
            x:               (B, n_genes) generatoroutput
            target_lib_size: (B,) each oftarget library size

        Returns:
             output expression matrix
        """
        eps = torch.finfo(x.dtype).eps
        sigmas = x.sum(dim=1)  # (B,)
        scale = target_lib_size / (sigmas + eps)  # (B,)
        return x * scale.unsqueeze(1)

    def forward(
        self,
        z: torch.Tensor,
        library_size: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.main(z)
        if self._output_activation == "relu":
            x = F.relu(x)
        elif self._output_activation == "tanh":
            x = torch.tanh(x)
        # "none" / "identity" -> no activation

        # LSN: Library Size Normalization following the output_lsn convention
        if self.lsn_lib_size is not None and library_size is not None:
            x = self._apply_lsn(x, library_size)

        return x


class ConditionalGenerator(nn.Module):
    """
    cscGAN-style conditiongenerator.

    Noise z receives a condition embedding, and each hidden layer applies
    condition-specific BatchNorm offset/scale followed by ReLU.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        gen_hidden: list = None,
        n_genes: int = 3000,
        n_conditions: int = 0,
        condition_dim: int = 32,
        use_batch_norm: bool = True,
        output_activation: str = "relu",
        lsn_lib_size: float = None,
    ):
        super().__init__()
        if n_conditions < 1:
            raise ValueError("ConditionalGenerator requires n_conditions >= 1")
        self.latent_dim = latent_dim
        self.n_genes = n_genes
        self.gen_hidden = gen_hidden or [256, 512]
        self.n_conditions = int(n_conditions)
        self.condition_dim = int(condition_dim)
        self._output_activation = output_activation
        self.lsn_lib_size = lsn_lib_size

        self.layers = nn.ModuleList()
        self.cond_norms = nn.ModuleList()
        in_dim = latent_dim
        for h_dim in self.gen_hidden:
            self.layers.append(nn.Linear(in_dim, h_dim, bias=not use_batch_norm))
            self.cond_norms.append(
                ConditionalBatchNorm1d(h_dim, self.n_conditions) if use_batch_norm else nn.Identity()
            )
            in_dim = h_dim
        self.output = nn.Linear(in_dim, n_genes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        library_size: torch.Tensor = None,
    ) -> torch.Tensor:
        if condition is None:
            raise ValueError("ConditionalGenerator.forward requires condition")
        condition = condition.long().view(-1).to(z.device)
        x = z
        for linear, norm in zip(self.layers, self.cond_norms):
            x = linear(x)
            if isinstance(norm, ConditionalBatchNorm1d):
                x = norm(x, condition)
            else:
                x = norm(x)
            x = F.relu(x)
        x = self.output(x)
        if self._output_activation == "relu":
            x = F.relu(x)
        elif self._output_activation == "tanh":
            x = torch.tanh(x)
        if self.lsn_lib_size is not None and library_size is not None:
            x = Generator._apply_lsn(x, library_size)
        return x


# ============================================================
# Critic (Discriminator)
# ============================================================

class Critic(nn.Module):
    """
    non-condition (Critic):compute Wasserstein  .

     : n_genes -> dis_hidden[0] -> ... -> dis_hidden[-1] -> 1
    hidden layer in each block: Linear -> ReLU; BatchNorm supports stable gradient-penalty training
    output layer:     Linear -> 1 ( activation)
    """

    def __init__(
        self,
        n_genes: int = 3000,
        dis_hidden: list = None,
    ):
        """
        Args:
            n_genes:         input gene count
            dis_hidden:      units in each hidden layer,default [512, 256]
        """
        super().__init__()
        if dis_hidden is None:
            dis_hidden = [512, 256]
        self.n_genes = n_genes
        self.dis_hidden = dis_hidden

        layers = []
        in_dim = n_genes
        for h_dim in dis_hidden:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim

        # output layer: Linear -> 1
        layers.append(nn.Linear(in_dim, 1))
        self.main = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialize"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class ConditionalCritic(nn.Module):
    """
    cscGAN-style condition critic.

    The critic extracts a hidden feature with an MLP and combines the
    unconditional score with a condition projection and label-specific head.
    """

    def __init__(
        self,
        n_genes: int = 3000,
        dis_hidden: list = None,
        n_conditions: int = 0,
        condition_dim: int = 32,
        projection: bool = True,
    ):
        super().__init__()
        if n_conditions < 1:
            raise ValueError("ConditionalCritic requires n_conditions >= 1")
        self.n_genes = n_genes
        self.dis_hidden = dis_hidden or [512, 256]
        self.n_conditions = int(n_conditions)
        self.condition_dim = int(condition_dim)
        self.projection = bool(projection)

        layers = []
        in_dim = n_genes
        for h_dim in self.dis_hidden:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim
        self.features = nn.Sequential(*layers)
        self.condition_projection = nn.Embedding(self.n_conditions, in_dim)
        self.condition_bias = nn.Embedding(self.n_conditions, 1)
        self.output = nn.Linear(in_dim, 1) if self.projection else None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_bias.weight)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition is None:
            raise ValueError("ConditionalCritic.forward requires condition")
        condition = condition.long().view(-1).to(x.device)
        features = self.features(x)
        cond_weight = self.condition_projection(condition)
        label_score = (features * cond_weight).sum(dim=1, keepdim=True)
        if self.projection:
            return label_score + self.output(features)
        return label_score + self.condition_bias(condition)


# ============================================================
# gradient penalty
# ============================================================

def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    condition: torch.Tensor = None,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty .

    inreal samplesandgenerate samples ofrandom value computegradientof L2 norm,
     its  1 of .

    GP = E[(||gradient_x D(x)||2 - 1)^2]

    Args:
        critic:       
        real_samples: real samples (B, n_genes)
        fake_samples: generate samples (B, n_genes)

    Returns:
        gradient_penalty:  loss
    """
    batch_size = real_samples.size(0)
    device = real_samples.device

    # random value number alpha in [0, 1]
    alpha = torch.rand(batch_size, 1, device=device)
    #  togenedimension
    alpha = alpha.expand_as(real_samples)

    #  value: x = alpha*real + (1-alpha)*fake
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    #  for value ofoutput
    if condition is None:
        d_interpolates = critic(interpolates)
    else:
        d_interpolates = critic(interpolates, condition)

    # computegradient
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # L2 norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    #   (||gradient||2 - 1)^2
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


# ============================================================
# WGAN-GP  
# ============================================================

class WGAN_GP:
    """
    WGAN-GP model ,  Generator + Critic + training/ .
      TensorFlow,  PyTorch implementation.

    follows the published design:
        - Optimizer: AMSGrad (Adam with amsgrad=True)
        - LR decay:  exponential learning-rate decay (alpha_0 -> alpha_final)
        - LSN:       Library Size Normalization (optional)
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        gen_hidden: list = None,
        dis_hidden: list = None,
        lambda_gp: float = 10.0,
        lr_gen: float = 2e-4,
        lr_dis: float = 2e-4,
        n_critic: int = 5,
        device: str = "cuda",
        output_activation: str = "relu",
        use_lsn: bool = False,
        lsn_lib_size: float = None,
        lr_decay: bool = False,
        lr_final: float = None,
        max_steps: int = None,
        n_conditions: int = 0,
        condition_dim: int = 32,
        conditional: bool = False,
    ):
        """
        Args:
            n_genes:             gene count (input/output dimension)
            latent_dim:          noise dimension
            gen_hidden:          generator hidden layer
            dis_hidden:           hidden layer
            lambda_gp:           gradient penalty number
            lr_gen:              generator learning rate (alpha_0)
            lr_dis:               learning rate (alpha_0)
            n_critic:            critic updates per generator update
            device:              "cuda" / "cpu"
            output_activation:   generator output activation
            use_lsn:             is enable LSN
            lsn_lib_size:        LSN target library size (such as 20000)
            lr_decay:            is enable exponential learning-rate decay
            lr_final:            final learning rate (alpha_final, default lr_gen*0.1)
            max_steps:            training number (usein LR decay)
        """
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_lsn = use_lsn
        self.conditional = bool(conditional and int(n_conditions) > 1)
        self.n_conditions = int(n_conditions) if self.conditional else 0
        self.condition_dim = int(condition_dim)

        if gen_hidden is None:
            gen_hidden = [256, 512]
        if dis_hidden is None:
            dis_hidden = [512, 256]

        # LSN parameter
        _lsn_lib_size = lsn_lib_size if use_lsn else None

        if self.conditional:
            self.generator = ConditionalGenerator(
                latent_dim=latent_dim,
                gen_hidden=gen_hidden,
                n_genes=n_genes,
                n_conditions=self.n_conditions,
                condition_dim=self.condition_dim,
                output_activation=output_activation,
                lsn_lib_size=_lsn_lib_size,
            ).to(self.device)
            self.critic = ConditionalCritic(
                n_genes=n_genes,
                dis_hidden=dis_hidden,
                n_conditions=self.n_conditions,
                condition_dim=self.condition_dim,
            ).to(self.device)
        else:
            self.generator = Generator(
                latent_dim=latent_dim,
                gen_hidden=gen_hidden,
                n_genes=n_genes,
                output_activation=output_activation,
                lsn_lib_size=_lsn_lib_size,
            ).to(self.device)
            self.critic = Critic(
                n_genes=n_genes,
                dis_hidden=dis_hidden,
            ).to(self.device)

        # AMSGrad   (follows the published design)
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=lr_gen, betas=(0.5, 0.9), amsgrad=True
        )
        self.optimizer_D = torch.optim.Adam(
            self.critic.parameters(), lr=lr_dis, betas=(0.5, 0.9), amsgrad=True
        )

        # exponential learning-rate decay following the exponential_decay convention
        self._lr_decay = lr_decay
        self._scheduler_G = None
        self._scheduler_D = None
        if lr_decay and max_steps and max_steps > 0:
            if lr_final is None:
                lr_final = lr_gen * 0.1
            # exponential_decay: lr = alpha_0 * (alpha_final/alpha_0)^(step/max_steps)
            # etc in StepLR of gamma = (alpha_final/alpha_0)^(1/max_steps)
            gamma_g = (lr_final / lr_gen) ** (1.0 / max_steps)
            gamma_d = (lr_final / lr_dis) ** (1.0 / max_steps)
            self._scheduler_G = torch.optim.lr_scheduler.StepLR(
                self.optimizer_G, step_size=1, gamma=gamma_g
            )
            self._scheduler_D = torch.optim.lr_scheduler.StepLR(
                self.optimizer_D, step_size=1, gamma=gamma_d
            )

        self._step = 0

    def _sample_conditions(self, batch_size: int, real_conditions: torch.Tensor = None) -> torch.Tensor:
        if not self.conditional:
            return None
        if real_conditions is not None and real_conditions.numel() > 0:
            idx = torch.randint(0, real_conditions.shape[0], (batch_size,), device=self.device)
            return real_conditions[idx].long()
        return torch.randint(0, self.n_conditions, (batch_size,), device=self.device)

    def train_step(self, real_batch: torch.Tensor, conditions: torch.Tensor = None) -> dict:
        """
        single step WGAN-GP training:
        1. update Critic n_critic  
        2. update Generator 1  

        Args:
            real_batch: real samples (B, n_genes)

        Returns:
            dict: {"d_loss": ..., "g_loss": ..., "gp": ...}
        """
        batch_size = real_batch.size(0)
        real_batch = real_batch.to(self.device)
        if self.conditional:
            if conditions is None:
                raise ValueError("Conditional WGAN_GP.train_step requires conditions")
            conditions = conditions.to(self.device).long().view(-1)

        # ---- training Critic (n_critic  ) ----
        d_loss_total = 0.0
        gp_total = 0.0
        for _ in range(self.n_critic):
            self.optimizer_D.zero_grad()

            # generate sample (optional LSN)
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_conditions = conditions if self.conditional else None
            if self.use_lsn:
                # fromreal inrandomsample size astarget
                real_lib_sizes = real_batch.sum(dim=1)
                idx = torch.randint(0, batch_size, (batch_size,), device=self.device)
                fake_lib_sizes = real_lib_sizes[idx]
                with torch.no_grad():
                    if self.conditional:
                        fake_batch = self.generator(z, fake_conditions, library_size=fake_lib_sizes)
                    else:
                        fake_batch = self.generator(z, library_size=fake_lib_sizes)
            else:
                with torch.no_grad():
                    fake_batch = self.generator(z, fake_conditions) if self.conditional else self.generator(z)

            # Critic output
            if self.conditional:
                d_real = self.critic(real_batch, conditions)
                d_fake = self.critic(fake_batch, fake_conditions)
            else:
                d_real = self.critic(real_batch)
                d_fake = self.critic(fake_batch)

            # WGAN loss: E[D(fake)] - E[D(real)]
            d_loss = d_fake.mean() - d_real.mean()

            # gradient penalty
            gp = compute_gradient_penalty(
                self.critic,
                real_batch,
                fake_batch,
                condition=conditions if self.conditional else None,
            )
            d_loss = d_loss + self.lambda_gp * gp

            d_loss.backward()
            self.optimizer_D.step()

            d_loss_total += d_loss.item()
            gp_total += gp.item()

            # LR decay (each critic step)
            if self._scheduler_D is not None:
                self._scheduler_D.step()

        d_loss_avg = d_loss_total / self.n_critic
        gp_avg = gp_total / self.n_critic

        # ---- training Generator (1  ) ----
        self.optimizer_G.zero_grad()

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_conditions = self._sample_conditions(batch_size, conditions)
        if self.use_lsn:
            real_lib_sizes = real_batch.sum(dim=1)
            idx = torch.randint(0, batch_size, (batch_size,), device=self.device)
            fake_lib_sizes = real_lib_sizes[idx]
            if self.conditional:
                fake_batch = self.generator(z, fake_conditions, library_size=fake_lib_sizes)
            else:
                fake_batch = self.generator(z, library_size=fake_lib_sizes)
        else:
            fake_batch = self.generator(z, fake_conditions) if self.conditional else self.generator(z)
        d_fake = self.critic(fake_batch, fake_conditions) if self.conditional else self.critic(fake_batch)

        # Generator loss: -E[D(fake)]
        g_loss = -d_fake.mean()

        g_loss.backward()
        self.optimizer_G.step()

        # LR decay (each generator step)
        if self._scheduler_G is not None:
            self._scheduler_G.step()

        self._step += 1

        return {
            "d_loss": d_loss_avg,
            "g_loss": g_loss.item(),
            "gp": gp_avg,
        }

    def generate(
        self,
        n_samples: int,
        batch_size: int = 256,
        library_size_mean: float = None,
        library_size_std: float = None,
        condition: torch.Tensor | int = None,
    ) -> torch.Tensor:
        """
        generatesyntheticsample.

        Args:
            n_samples:           number of samples to generate
            batch_size:         inference batch size
            library_size_mean:  LSN target library sizemean (training-data statistics)
            library_size_std:   LSN target library sizestandard deviation (training-data statistics)

        Returns:
            (n_samples, n_genes) numpy array
        """
        self.generator.eval()
        samples = []

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                n = end - start
                z = torch.randn(n, self.latent_dim, device=self.device)
                if self.conditional:
                    if condition is None:
                        y = torch.randint(0, self.n_conditions, (n,), device=self.device)
                    elif torch.is_tensor(condition):
                        y = condition[start:end].to(self.device).long().view(-1)
                    else:
                        y = torch.full((n,), int(condition), device=self.device, dtype=torch.long)
                else:
                    y = None

                if self.use_lsn and library_size_mean is not None:
                    # positive distribution sample size (reference truncated normal approximation)
                    if library_size_std is not None and library_size_std > 0:
                        lib_sizes = torch.normal(
                            mean=library_size_mean,
                            std=library_size_std,
                            size=(n,),
                        ).clamp(min=1.0).to(self.device)
                    else:
                        lib_sizes = torch.full((n,), library_size_mean, device=self.device)
                    fake = self.generator(z, y, library_size=lib_sizes) if self.conditional else self.generator(z, library_size=lib_sizes)
                else:
                    fake = self.generator(z, y) if self.conditional else self.generator(z)
                samples.append(fake.cpu())

        self.generator.train()
        return torch.cat(samples, dim=0).numpy()

    def save(self, path: str, extra: dict = None):
        """savemodel weights"""
        state = {
            "generator": self.generator.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "step": self._step,
            "n_genes": self.n_genes,
            "latent_dim": self.latent_dim,
            "gen_hidden": self.generator.gen_hidden,
            "dis_hidden": self.critic.dis_hidden,
            "use_lsn": self.use_lsn,
            "lsn_lib_size": self.generator.lsn_lib_size,
            "conditional": self.conditional,
            "n_conditions": self.n_conditions,
            "condition_dim": self.condition_dim,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load(self, path: str):
        """loadmodel weights"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(ckpt["generator"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer_G.load_state_dict(ckpt["optimizer_G"])
        self.optimizer_D.load_state_dict(ckpt["optimizer_D"])
        self._step = ckpt.get("step", 0)

    def train(self):
        """ as training mode"""
        self.generator.train()
        self.critic.train()

    def eval(self):
        """ asevaluatemode"""
        self.generator.eval()
        self.critic.eval()

    @property
    def step(self) -> int:
        return self._step
