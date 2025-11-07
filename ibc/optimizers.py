from __future__ import annotations

import dataclasses
import enum
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================================== #
# Model optimization.
# =================================================================== #


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    lr_scheduler_step: int = 100
    lr_scheduler_gamma: float = 0.99


# =================================================================== #
# Stochastic optimization for EBM training and inference.
# =================================================================== #


@dataclasses.dataclass
class StochasticOptimizerConfig:
    bounds: np.ndarray
    """Bounds on the samples, min/max for each dimension."""

    iters: int
    """The total number of inference iters."""

    infer_samples: int
    negative_samples: int
    """The number of counter-examples to sample per iter during training."""

    inference_samples: int
    """The number of candidates to sample per iter during inference."""


class StochasticOptimizer(Protocol):
    """Functionality that needs to be implemented by all stochastic optimizers."""

    device: torch.device

    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        """Sample counter-negatives for feeding to the InfoNCE objective."""

    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action conditioned on the current observation."""


@dataclasses.dataclass
class DerivativeFreeConfig(StochasticOptimizerConfig):
    noise_scale: float = 0.33
    noise_shrink: float = 0.5
    iters: int = 3
    infer_samples: int = 256    # 采样数量
    negative_samples: int = 256  # 负样本数量
    inference_samples: int = 2 ** 14


@dataclasses.dataclass
class DerivativeFreeOptimizer:
    """A simple derivative-free optimizer. Great for up to 5 dimensions."""

    device: torch.device
    noise_scale: float
    noise_shrink: float
    iters: int
    infer_samples: int
    negative_samples: int
    inference_samples: int
    bounds: np.ndarray

    @staticmethod
    def initialize(
        config: DerivativeFreeConfig, device_type: str
    ) -> DerivativeFreeOptimizer:
        return DerivativeFreeOptimizer(
            device=torch.device(device_type if torch.cuda.is_available() else "cpu"),
            noise_scale=config.noise_scale,
            noise_shrink=config.noise_shrink,
            iters=config.iters,
            infer_samples=config.infer_samples,
            negative_samples=config.negative_samples,
            inference_samples=config.inference_samples,
            # Ensure bounds are float32 to avoid dtype mismatches with model FloatTensors
            bounds=config.bounds.astype("float32") if isinstance(config.bounds, np.ndarray) else config.bounds,
        )

    # def _sample(self, num_samples: int) -> torch.Tensor:
    #     """Helper method for drawing samples from the uniform random distribution."""
    #     size = (num_samples, self.bounds.shape[1])
    #     samples = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)
    #     # Use float32 numpy -> torch conversion to match model dtype (float32)
    #     return torch.as_tensor(samples.astype(np.float32), dtype=torch.float32, device=self.device)
    
    def _sample(self, num_samples: int) -> torch.Tensor:
        """支持三维数据的均匀分布采样（样本形状：(num_samples, d1, d2)）"""

        # 计算样本形状：(num_samples, d1, d2)，其中 (d1, d2) 是 self.bounds 除第一维外的形状
        size = (num_samples,) + self.bounds.shape[1:]  # 关键修改：用 self.bounds 的后几维作为数据维度
        
        # 从三维边界中采样：low 和 high 是 (d1, d2) 形状，与样本的空间维度匹配
        samples = np.random.uniform(
            low=self.bounds[0, ...],  # 三维下界：shape=(d1, d2)
            high=self.bounds[1, ...],  # 三维上界：shape=(d1, d2)
            size=size  # 最终样本形状：(num_samples, d1, d2)
        )
    
        # 转换为张量
        samples_tensor = torch.as_tensor(
            samples.astype(np.float32),
            dtype=torch.float32,
            device=self.device
        )
        # print(f"样本形状：{samples_tensor.shape}")
        
        # 转换为 float32 张量并返回
        return torch.as_tensor(
            samples.astype(np.float32), 
            dtype=torch.float32, 
            device=self.device
        )

    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        del ebm  # The derivative-free optimizer does not use the ebm for sampling.
        samples = self._sample(batch_size * self.infer_samples)
        return samples.reshape(batch_size, self.infer_samples, -1)
    
    def negative_sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        del ebm  # The derivative-free optimizer does not use the ebm for sampling.
        samples = self._sample(batch_size * self.negative_samples)
        return samples.reshape(batch_size, self.negative_samples, -1)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action given a trained EBM."""
        noise_scale = self.noise_scale
        # Make sure bounds tensor matches the dtype of the samples and model (float32)
        bounds = torch.as_tensor(self.bounds, dtype=torch.float32, device=self.device)

        samples = self._sample(x.size(0) * self.infer_samples)
        samples = samples.reshape(x.size(0), self.infer_samples, -1)

        for i in range(self.iters):
            # Compute energies.
            energies = ebm(x, samples)
            probs = F.softmax(-energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self.infer_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=bounds[0, :], max=bounds[1, :])

            noise_scale *= self.noise_shrink

        # Return target with highest probability.
        energies = ebm(x, samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs, :]
    
    @torch.no_grad()
    def negative_infer(self, x: torch.Tensor, progress, ebm: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        # Curriculum: start with random, gradually increase difficulty
        sampling_config = self._get_sampling_config(progress)

        current_iters = sampling_config['iters']
        current_noise_scale = sampling_config['noise_scale']
        current_noise_shrink = sampling_config['noise_shrink']

        bounds = torch.as_tensor(self.bounds, dtype=torch.float32, device=self.device)

        # Initial negative samples: (B, N_neg, act_dim)
        samples = self.negative_sample(x.size(0), ebm)

        # Early training: use random negatives (iters=0)
        if current_iters == 0:
            energies = ebm(x, samples)
            return samples, energies

        for i in range(current_iters):
            # Compute energy for all current negative samples
            energies = ebm(x, samples)  # (B, N_neg)

            # Select hard negatives: low energy but still negatives
            # Temperature controls hardness: higher T = softer selection
            temperature = 1.0 + (1 - progress) * 2.0  # T: 3.0 → 1.0
            probs = F.softmax(-energies / temperature, dim=-1)

            # Resample indices based on energy-based probabilities
            idxs = torch.multinomial(probs, self.negative_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

            # Add Gaussian exploration noise and clamp
            samples = samples + torch.randn_like(samples) * current_noise_scale
            samples = samples.clamp(min=bounds[0, :], max=bounds[1, :])

            current_noise_scale *= current_noise_shrink

        # Final energy evaluation for all negative samples
        energies = ebm(x, samples)  # (B, N_neg)

        return samples, energies

    def _get_sampling_config(self, progress):
        """
        根据训练进度返回采样配置。

        Curriculum strategy:
        - Early (0-30%): Random negatives to learn basic discrimination
        - Mid (30-70%): Medium difficulty with some refinement
        - Late (70-100%): Hard negatives with full refinement
        """
        if progress < 0.15:  # Early training: random to easy negatives
            return {
                'iters': 0 if progress < 0.1 else max(1, self.iters // 3),
                'noise_scale': 0.0 if progress < 0.1 else self.noise_scale * 1.5,
                'noise_shrink': 1.0 if progress < 0.1 else 0.7,
                'difficulty': 'easy'
            }
        elif progress < 0.6:  # Mid training: medium difficulty
            return {
                'iters': max(1, self.iters // 2),
                'noise_scale': self.noise_scale,
                'noise_shrink': self.noise_shrink,
                'difficulty': 'medium'
            }
        else:  # Late training: hard negatives
            return {
                'iters': self.iters,
                'noise_scale': self.noise_scale * 0.5,
                'noise_shrink': 0.9,
                'difficulty': 'hard'
            }



class StochasticOptimizerType(enum.Enum):
    # Note(kevin): The paper describes three types of samplers. Right now, we just have
    # the derivative free sampler implemented.
    DERIVATIVE_FREE = enum.auto()


if __name__ == "__main__":
    from data.dataset import CoordinateRegression, DatasetConfig

    dataset = CoordinateRegression(DatasetConfig(dataset_size=10))
    bounds = dataset.get_target_bounds()

    config = DerivativeFreeConfig(bounds=bounds, train_samples=256)
    so = DerivativeFreeOptimizer.initialize(config, "cuda")

    negatives = so.sample(64, nn.Identity())
    assert negatives.shape == (64, config.train_samples, bounds.shape[1])
