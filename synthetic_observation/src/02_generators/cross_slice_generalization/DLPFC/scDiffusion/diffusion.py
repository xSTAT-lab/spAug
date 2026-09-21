"""Gaussian diffusion process used by the project-local scDiffusion model."""

import math
import numpy as np
import torch


# ============================================================
# Beta schedule
# ============================================================

def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> np.ndarray:
    """
    get definitionof beta schedule.

    Args:
        schedule_name: "linear" or "cosine"
        num_diffusion_timesteps: number of diffusion steps T

    Returns:
        (T,) numpy array of betas
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.minimum(
            np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64),
            0.999,
        )
    elif schedule_name == "cosine":
        return _betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def _betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar_fn, max_beta=0.999):
    """  alpha  numbergenerate beta schedule."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return np.array(betas)


# ============================================================
# helper functions
# ============================================================

def _extract_into_tensor(arr: np.ndarray, timesteps: torch.Tensor, broadcast_shape) -> torch.Tensor:
    """
    Extract values from a 1-D NumPy array using index positions.

    Args:
        arr:             (T,) numpy array
        timesteps:       (B,) index 
        broadcast_shape: targetshape (B, ...)

    Returns:
         afterof 
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# ============================================================
# Gaussian diffusion process
# ============================================================

class GaussianDiffusion:
    """
    DDPM Gaussian diffusion process.

    beforeto: q(x_t|x_0) = N(x_t; sqrtabar_t x_0, (1-abar_t) I)
     to: p_theta(x_{t-1}|x_t) = N(x_{t-1}; mu_theta(x_t, t), sum_theta(x_t, t))

    model-predicted noise epsilon using MSE loss.

    Args:
        betas:            (T,) beta schedulearray
        model_mean_type:  "epsilon" (predicted noise) or "x_start" (predict x_0)
        model_var_type:   "fixed_small" or "fixed_large"
        loss_type:        "mse"
        rescale_timesteps: is timestepto [0, 1000]
    """

    def __init__(
        self,
        *,
        betas: np.ndarray,
        model_mean_type: str = "epsilon",
        model_var_type: str = "fixed_small",
        loss_type: str = "mse",
        rescale_timesteps: bool = False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        #  precomputed diffusion parameters
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # q(x_t|x_0) parameter
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1}|x_t, x_0) posterior parameters
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    # ============================================================
    # beforeto  q(x_t | x_0)
    # ============================================================

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None) -> torch.Tensor:
        """
        forward noising: x_t = sqrtabar_t * x_0 + sqrt(1-abar_t) * epsilon

        Args:
            x_start: (B, G) clean data
            t:       (B,)   timestep
            noise:   (B, G) optional noise

        Returns:
            (B, G)  data
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ============================================================
    # posterior parameters
    # ============================================================

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """compute q(x_{t-1}|x_t, x_0) mean and variance."""
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    # ============================================================
    #  to : modeloutput -> distributionparameter
    # ============================================================

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """fromnoise epsilon   x_0."""
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """from x_0  noise epsilon."""
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_mean_variance(self, model, x, t, clip_denoised=True, model_kwargs=None):
        """
        compute p(x_{t-1}|x_t) mean and variance.

        Args:
            model:         denoising network,   model(x, t) -> prediction
            x:             (B, G) current x_t
            t:             (B,)   timestep
            clip_denoised: is  x_0 predictto [-1, 1]
            model_kwargs:   parameter

        Returns:
            dict: mean, variance, log_variance, pred_xstart
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        # modelcall: model(x_t, t.unsqueeze(1))
        model_output = model(x.float(), self._scale_timesteps(t).unsqueeze(1), **model_kwargs)

        # variance (fixed)
        if self.model_var_type == "fixed_large":
            model_variance = _extract_into_tensor(
                np.append(self.posterior_variance[1], self.betas[1:]), t, x.shape
            )
            model_log_variance = _extract_into_tensor(
                np.log(np.append(self.posterior_variance[1], self.betas[1:])), t, x.shape
            )
        else:  # fixed_small
            model_variance = _extract_into_tensor(self.posterior_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )

        # from model output to x_0
        def process_xstart(pred):
            if clip_denoised:
                return pred.clamp(-1, 1)
            return pred

        if self.model_mean_type == "epsilon":
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x, t, model_output))
        elif self.model_mean_type == "x_start":
            pred_xstart = process_xstart(model_output)
        else:
            raise NotImplementedError(f"unknown model_mean_type: {self.model_mean_type}")

        # compute p(x_{t-1}|x_t) mean
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    # ============================================================
    # DDPM sample
    # ============================================================

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, cond_fn=None):
        """
        single step DDPM sample: x_t -> x_{t-1}.

        Returns:
            dict: sample, pred_xstart
        """
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        model_mean = out["mean"]
        if cond_fn is not None:
            gradient = cond_fn(x, t, out)
            model_mean = model_mean + gradient
        noise = torch.randn_like(x)
        # t=0  
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self, model, shape, noise=None, clip_denoised=True,
        model_kwargs=None, device=None, progress=False, cond_fn=None,
    ):
        """
        DDPM  tosample: from noise x_T  to x_0.

        Args:
            model: denoising network
            shape: (N, G) sampleshape
            noise: optional noise
            clip_denoised:   x_0 predict
            device: computedevice
            progress:  progress

        Returns:
            (N, G) tensor generate samples
        """
        if device is None:
            device = next(model.parameters()).device

        if noise is not None:
            x = noise
        else:
            x = torch.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            if cond_fn is None:
                with torch.no_grad():
                    out = self.p_sample(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
            else:
                out = self.p_sample(
                    model, x, t, clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs, cond_fn=cond_fn,
                )
                out = {key: value.detach() if torch.is_tensor(value) else value for key, value in out.items()}
            x = out["sample"]

        return x

    # ============================================================
    # DDIM accelerated sampling
    # ============================================================

    def ddim_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, eta=0.0, cond_fn=None):
        """
        single step DDIM sample (deterministic, eta=0  as ODE).

        Returns:
            dict: sample, pred_xstart
        """
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        if cond_fn is not None:
            gradient = cond_fn(x, t, out)
            eps = eps - torch.sqrt(1 - alpha_bar) * gradient
        sigma = (
            eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self, model, shape, noise=None, clip_denoised=True,
        model_kwargs=None, device=None, progress=False, eta=0.0, cond_fn=None,
    ):
        """
        DDIM accelerated sampling: availablemore numbergenerate.

        Returns:
            (N, G) tensor
        """
        if device is None:
            device = next(model.parameters()).device

        if noise is not None:
            x = noise
        else:
            x = torch.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            if cond_fn is None:
                with torch.no_grad():
                    out = self.ddim_sample(model, x, t, clip_denoised=clip_denoised,
                                           model_kwargs=model_kwargs, eta=eta)
            else:
                out = self.ddim_sample(
                    model, x, t, clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs, eta=eta, cond_fn=cond_fn,
                )
                out = {key: value.detach() if torch.is_tensor(value) else value for key, value in out.items()}
            x = out["sample"]

        return x

    # ============================================================
    # training loss
    # ============================================================

    def training_losses(self, model, x_start, t, noise=None):
        """
        compute training loss (MSE).

        Args:
            model:   denoising network
            x_start: (B, G) clean data
            t:       (B,)   timestep
            noise:   optional noise

        Returns:
            dict: {"loss": scalar}
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)

        # modelpredict
        model_output = model(x_t.float(), self._scale_timesteps(t).unsqueeze(1))

        # target
        if self.model_mean_type == "epsilon":
            target = noise
        elif self.model_mean_type == "x_start":
            target = x_start
        else:
            raise NotImplementedError

        # MSE loss (fornon- batch dimension mean,  for batch  mean)
        loss = ((target - model_output) ** 2).mean()
        return {"loss": loss}
