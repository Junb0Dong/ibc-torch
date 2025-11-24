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
    iters: int
    infer_samples: int
    negative_samples: int    


class StochasticOptimizer(Protocol):
    """Functionality that needs to be implemented by all stochastic optimizers."""

    device: torch.device

    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        """Sample counter-negatives for feeding to the InfoNCE objective."""

    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action conditioned on the current observation."""


@dataclasses.dataclass
class DerivativeFreeConfig(StochasticOptimizerConfig):
    noise_scale: float = 0.1
    noise_shrink: float = 0.9
    iters: int = 3
    infer_samples: int = 256    # 采样数量
    negative_samples: int = 256  # 负样本数量



@dataclasses.dataclass
class LangevinMCMCConfig(StochasticOptimizerConfig):
    noise_scale: float = 0.1
    noise_shrink: float = 0.99
    iters: int = 25
    infer_samples: int = 4096
    negative_samples: int = 256
    # apply to langevin mcmc
    grad_clip: float = 1.0
    delta_action_clip: float = 1
    stepsize: float = 1
    min_actions: float = -1
    max_actions: float = 1
    grad_norm_type: str = "l1"
    use_polynomial_rate: bool = True
    stepsize_power: float = 2.0
    stepsize_final: float = 1e-5


@dataclasses.dataclass
class DerivativeFreeOptimizer:
    """A simple derivative-free optimizer. Great for up to 5 dimensions."""

    device: torch.device
    noise_scale: float
    noise_shrink: float
    iters: int
    infer_samples: int
    negative_samples: int
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
            # samples_pos = torch.cat([samples, agent_pos.repeat(1, samples.size(1), 1)], dim=-1)
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
        # samples_pos = torch.cat([samples, agent_pos.unsqueeze(1).repeat(1, samples.size(1), 1)], dim=-1)
        energies = ebm(x, samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs, :]
    
    @torch.no_grad()
    def negative_infer(self, x: torch.Tensor, progress, ebm: nn.Module, agent_pos) -> tuple[torch.Tensor, torch.Tensor]:
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
            samples_pos = torch.cat([samples, agent_pos.unsqueeze(1).repeat(1, samples.size(1), 1)], dim=-1)
            energies = ebm(x, samples_pos)
            return samples, energies

        for i in range(current_iters):
            # Compute energy for all current negative samples
            samples_pos = torch.cat([samples, agent_pos.unsqueeze(1).repeat(1, samples.size(1), 1)], dim=-1)
            energies = ebm(x, samples_pos)  # (B, N_neg)

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
        samples_pos = torch.cat([samples, agent_pos.unsqueeze(1).repeat(1, samples.size(1), 1)], dim=-1)
        energies = ebm(x, samples_pos)  # (B, N_neg)

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

@dataclasses.dataclass
class LangevinMCMCOptimizer(StochasticOptimizer):
    
    device: torch.device
    noise_scale: float
    noise_shrink: float
    iters: int
    infer_samples: int
    negative_samples: int
    bounds: np.ndarray
    grad_clip: float
    delta_action_clip: float
    stepsize: float
    min_actions: float
    max_actions: float
    grad_norm_type: str
    use_polynomial_rate: bool
    stepsize_power: float
    stepsize_final: float
    @staticmethod
    def initialize(
        config: LangevinMCMCConfig, device_type: str
    ) -> LangevinMCMCOptimizer:
        return LangevinMCMCOptimizer(
            device=torch.device(device_type if torch.cuda.is_available() else "cpu"),
            noise_scale=config.noise_scale,
            noise_shrink=config.noise_shrink,
            iters=config.iters,
            infer_samples=config.infer_samples,
            negative_samples=config.negative_samples,
            # Ensure bounds are float32 to avoid dtype mismatches with model FloatTensors
            bounds=config.bounds.astype("float32") if isinstance(config.bounds, np.ndarray) else config.bounds,
            delta_action_clip = config.delta_action_clip,
            grad_clip = config.grad_clip,
            stepsize = config.stepsize,
            min_actions = config.min_actions,
            max_actions = config.max_actions,
            grad_norm_type=config.grad_norm_type,
            use_polynomial_rate=config.use_polynomial_rate,
            stepsize_power=config.stepsize_power,
            stepsize_final=config.stepsize_final,
        )

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

        # 转换为 float32 张量并返回
        return torch.as_tensor(
            samples.astype(np.float32), 
            dtype=torch.float32, 
            device=self.device
        )

    def _get_schedule(self):
        """获取步长调度器"""
        if self.use_polynomial_rate:
            return self._polynomial_schedule
        else:
            return self._exponential_schedule
    
    def _polynomial_schedule(self, step_index: int) -> float:
        """多项式步长衰减"""
        if self.iters <= 1:
            return self.stepsize
        progress = step_index / (self.iters - 1)
        return (self.stepsize - self.stepsize_final) * ((1 - progress) ** self.stepsize_power) + self.stepsize_final
    
    def _exponential_schedule(self, step_index: int) -> float:
        """指数步长衰减"""
        return self.stepsize * (self.noise_shrink ** step_index)
    
    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        del ebm  # The derivative-free optimizer does not use the ebm for sampling.
        samples = self._sample(batch_size * self.infer_samples)
        return samples.reshape(batch_size, self.infer_samples, -1)
    
    def negative_sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        del ebm  # The derivative-free optimizer does not use the ebm for sampling.
        samples = self._sample(batch_size * self.negative_samples)
        return samples.reshape(batch_size, self.negative_samples, -1)
    
    def negative_sample_delta(self, batch_size: int, ebm: nn.Module, delta_samples: int) -> torch.Tensor:
        del ebm  # The derivative-free optimizer does not use the ebm for sampling.
        samples = self._sample(batch_size * (self.negative_samples + delta_samples))
        return samples.reshape(batch_size, (self.negative_samples + delta_samples), -1)
    
    def gradient_wrt_act(self, energy_network, observations, actions):
        # 传入的actions可能已有grad，但clone.detach确保独立计算图
        actions = actions.clone().detach().requires_grad_(True)
        energies = energy_network(observations, actions)

        denergies_dactions = torch.autograd.grad(
            energies.sum(),
            actions,
            create_graph=False,  # 修复OOM：采样无需二阶图
            retain_graph=False   # 无需保留，每个step独立
        )[0]

        return denergies_dactions, energies

    
    def compute_grad_norm(self, de_dact):
        """Given de_dact and the type, compute the norm."""
        if self.grad_norm_type is not None:
            grad_norm_type_to_ord = {
                'l1': 1,
                'l2': 2,
                'inf': float('inf')
            }
            grad_type = grad_norm_type_to_ord[self.grad_norm_type]
            # 计算每个样本的梯度范数（沿动作维度）
            grad_norms = torch.norm(de_dact, p=grad_type, dim=1)
        else:
            # 修复：避免[:,0]维度问题，用size(0)创建batch大小的零向量
            grad_norms = torch.zeros(de_dact.size(0), dtype=de_dact.dtype, device=de_dact.device)
        
        return grad_norms
    
    def langevin_step(self, energy_network, observations, actions, noise_scale):
        l_lambda = 1.0
        
        de_dact, energies = self.gradient_wrt_act(energy_network, observations, actions)
        scaled_delta_clip = self.delta_action_clip * 0.5 * (self.max_actions - self.min_actions)
        grad_norms = self.compute_grad_norm(de_dact.detach())

        if self.grad_clip is not None:
            de_dact = torch.clamp(de_dact, -self.grad_clip, self.grad_clip)

        gradient_scale = 0.5
        noise = torch.randn_like(actions)

        with torch.no_grad():
            de_dact = (gradient_scale * l_lambda * de_dact +
                    noise * l_lambda * noise_scale)

            delta_actions = torch.clamp(
                self.stepsize * de_dact,
                -scaled_delta_clip,
                scaled_delta_clip
            )

            updated_actions = torch.clamp(
                actions - delta_actions,
                self.min_actions,
                self.max_actions
            )

        return {
            'action': updated_actions,
            'energies': energies,          # DO NOT DETACH!
            'grad_norms': grad_norms.detach()
        }

        
    def langevin_optimize(self, energy_network: nn.Module, observations: torch.Tensor,
                        action_samples: torch.Tensor, return_chain: bool = False):
        """
        Main loop.
        - action_samples expected shape: (B * num_samples, action_dim) 或 (B, num_samples, action_dim) 需在调用方统一。
        - 内部将使用 flat shape (b_times_n, action_dim) 进行计算。
        """

        actions = action_samples.clone().detach().requires_grad_(True)  # 创建一个“独立，无历史依赖，可求导”的张量
        local_noise_scale = float(self.noise_scale)
        schedule = self._get_schedule()
        chain_data = None

        for step in range(self.iters):
            self.stepsize = schedule(step)
            result = self.langevin_step(
                energy_network,
                observations,
                actions,
                local_noise_scale
            )
            # detach+clone释放前step内存（图被丢弃）
            old_actions = actions  # 可选：显式del
            actions = result['action'].clone().detach().requires_grad_(True)    # 切断/隔离历史计算图。赋予干净的E/a的计算图
            del old_actions  # 辅助GC

            local_noise_scale *= float(self.noise_shrink)
            result['action'] = result['action'].detach()
            
        # 重新进行一次前向计算，创建loss的计算图，避免对chain_data的进行二次求导
        final_actions = result['action']
        final_energies_for_loss = energy_network(observations, final_actions) # (B, N_neg)
        result['energies'] = final_energies_for_loss
        result['action'] = final_actions.detach()
        
        return (result, chain_data) if return_chain else (result, None)


    def get_min_energy_action(self, energy_network, observations, actions, batch_size=None, num_samples=None):
        """
        选取每个 batch 的最低能量动作。
        要求：
        - 如果 actions 是 (B*num_samples, D)，需传入 batch_size, num_samples 以便 reshape。
        - 如果 actions 已是 (B, K, D)，则可直接使用。
        返回 shape: (B, D)
        """
        # 规范化 actions/energies 的形状以便索引
        if actions.dim() == 2 and batch_size is not None and num_samples is not None:
            # flat -> (B, K, D)
            B = batch_size
            K = num_samples
            D = actions.shape[-1]
            actions_reshaped = actions.view(B, K, D)
            energies = energy_network(observations, actions).view(B, K)
        elif actions.dim() == 3:
            actions_reshaped = actions
            energies = energy_network(observations, actions)
        else:
            raise ValueError("请确保传入正确的 actions 形状，或提供 batch_size 和 num_samples.")

        min_indices = torch.argmin(energies, dim=1)
        batch_indices = torch.arange(energies.size(0), device=energies.device)
        min_energy_actions = actions_reshaped[batch_indices, min_indices]
        return min_energy_actions

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

