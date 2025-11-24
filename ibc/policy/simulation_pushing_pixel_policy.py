from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ibc.model.common.normalizer import LinearNormalizer
from ibc.policy.base_image_policy import BaseImagePolicy
from ibc import models, optimizers
import numpy as np

from typing import List

"""
这个policy针对ibc的pixel来配置，包括参数和网络的设置，训练的loss计算以及预测动作。
"""
# TODO：对时序信息进行处理，在通道上进行堆叠，但是输出动作只需要最后一个动作即可
# 模型基于长度为 sequence_length 的连续观测序列，输出单步动作
class SimulationPushingPixelPolicy(BaseImagePolicy):
    def __init__(self,
        shape_meta: dict,
        input_channels: int,
        cnn_blocks: List[int],
        n_action_steps: int,
        resnet_width: int,
        resnet_num_blocks: int,
        target_bounds:  torch.Tensor,
        # stochastic_optimizer_train_samples: int,
        infer_samples: int,
        negative_samples: int,
        # uniform_boundary_buffer: float,
        optimize_again: bool = False,
        device_type: str = "cuda",
        infer_iter: int = 25,
        use_polynomial_rate: bool = True,
        ):
        super().__init__()
        _device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        print(f"Using device: {_device}")
        self._device = _device
        self.n_action_steps = n_action_steps
        print("n_action_steps:", n_action_steps)
        
        self.optimize_again = optimize_again
        
        action_dim = shape_meta['action']['shape']
        
        # print([v['shape'] for v in shape_meta['obs'].values()])
        # print(sum(np.prod(v['shape']) for v in shape_meta['obs'].values()))
        # input_dim = sum(np.prod(v['shape']) for v in shape_meta['obs'].values())
        # input_dim = input_dim + np.prod(action_dim)
        # print("input_dim:", input_dim)
        
        # 使用ConvMaxPool加DenseResMLP
        model = models.ConvResnetEBM(
            cnn_in_channels=input_channels,
            cnn_blocks = cnn_blocks,
            resnet_width=resnet_width,
            resnet_num_blocks=resnet_num_blocks,
            action_dim=action_dim,
        )

        print("ConvResnetEBM model:", model)
        model.to(_device)
        self.model = model

        # TODO: 动态获取target_bounds，应该要在数据集的代码中获取，额外添加函数`get_target_bounds()`
        # 应该在workspace中获取
        print("the type of target_bounds:", type(target_bounds)) 
        target_bounds = torch.tensor(target_bounds, dtype=torch.float32)
        stochastic_optim_config = optimizers.LangevinMCMCConfig(
            bounds=target_bounds,
            infer_samples=infer_samples,
            negative_samples=negative_samples,
            iters=infer_iter,
            use_polynomial_rate=use_polynomial_rate
        )

        self.stochastic_optimizer = optimizers.LangevinMCMCOptimizer.initialize(
            stochastic_optim_config,
            device_type,
        )

        self.normalizer = LinearNormalizer()

    def predict_action( self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # normalized_obs = {}
        # for key, value in obs_dict.items():
        #     normalized_obs[key] = self.normalizer[key].normalize(value)

        # obs_input = torch.cat([normalized_obs[k] for k in obs_dict], dim=-1)
        
        obs_input = obs_dict['image'][:, -1]
        
        sample_actions = self.stochastic_optimizer.sample(obs_input.size(0), self.model)
        
        # 临时启用梯度：包围 Langevin 调用
        with torch.enable_grad():
            langevin_result, _ = self.stochastic_optimizer.langevin_optimize(
                energy_network=self.model, 
                observations=obs_input.to(self._device), 
                action_samples=sample_actions
            )
        
        if self.optimize_again:
            # 第二次也需启用梯度（pre_actions.requires_grad_ 仅标记，但需图构建）
            pre_actions = langevin_result['action']
            pre_actions.requires_grad_(True)  # 保持，但无效无图
            with torch.enable_grad():  # 再次包围
                langevin_result, _ = self.stochastic_optimizer.langevin_optimize(
                    energy_network=self.model, 
                    observations=obs_input.to(self._device), 
                    action_samples=pre_actions
                )
            
        prediction_action = self.stochastic_optimizer.get_min_energy_action(self.model, obs_input.to(self._device), langevin_result['action'])

        action_tensor = self.normalizer['action'].unnormalize(prediction_action)
        action_tensor = action_tensor.unsqueeze(1).repeat(1, self.n_action_steps, 1)

        return {
            'action': action_tensor,
            'action_pred': action_tensor
        }
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        将传入的 normalizer 对象的状态（即其内部的参数）复制到当前策略类实例的 self.normalizer 中
        """
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        normalized_obs = {}
        for key, value in batch['obs'].items():
            normalized_obs[key] = self.normalizer[key].normalize(value)

        obs_input = torch.cat([normalized_obs[k] for k in batch['obs']], dim=-1)

        target = self.normalizer['action'].normalize(batch['action'])
        target = target.squeeze(1)

        negatives = self.stochastic_optimizer.negative_sample(obs_input.size(0), self.model)
        negatives = self.normalizer['action'].normalize(negatives)

        targets = torch.cat([target.unsqueeze(dim=1), negatives], dim=1)
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(self._device)

        energy = self.model(obs_input, targets)
        logits = -1.0 * energy
        loss = F.cross_entropy(logits, ground_truth)
        return loss


    # TODO：添加last_action信息
    # TODO： obs的时间信息堆叠到通道上
    # def InfoNCE_loss(self, batch, progress, training=True):
    #     obs_image = batch['obs']['image'].squeeze(1)
    #     B, C, H, W = obs_image.shape

    #     # Positive samples
    #     positive = self.normalizer['action'].normalize(batch['action'])
    #     E_pos = self.model(obs_image, positive)  # (B, 1)

    #     negative = self.stochastic_optimizer.negative_sample(B, self.model)
        
    #     with torch.enable_grad():
    #         langevin_result, _ = self.stochastic_optimizer.langevin_optimize(
    #                 energy_network = self.model, # 使用 model 本身，但只允许 E_neg 梯度通过
    #                 observations = obs_image.detach(), # 避免 obs 的梯度也进入 MCMC 图
    #                 action_samples = negative
    #             )

    #     E_neg = langevin_result['energies']  # (B, N_neg)
    #     if E_neg.mean() < E_pos.mean():
    #         print(f"[Warning] E_neg ({E_neg.mean().item():.3f}) < E_pos ({E_pos.mean().item():.3f}) at progress={progress:.2f}")
    #     E_all = torch.cat([E_pos, E_neg], dim=1)  # (B, N_neg+1)

    #     # InfoNCE loss using stable logsumexp
    #     logits = -E_all  # (B, N_neg+1)
    #     log_sum_exp = torch.logsumexp(logits, dim=1, keepdim=True)  # (B, 1)
    #     log_prob_pos = logits[:, 0:1] - log_sum_exp  # (B, 1)

    #     # 添加正则化
    #     l2_reg_lambda = getattr(self, 'l2_reg_lambda', 1e-4)  # 1e-5 ~ 1e-3 可调
    #     l2_reg_loss = 0.0
    #     for param in self.model.parameters():
    #         if param.requires_grad:
    #             l2_reg_loss += torch.norm(param, p=2)  # L2范数惩罚
    #     l2_reg_loss = l2_reg_lambda * l2_reg_loss

    #     # Negative log likelihood
    #     loss = -log_prob_pos.mean() + l2_reg_loss

    #     return loss


    def InfoNCE_loss(self, batch, progress, training=True):
        obs_image = batch['obs']['image'].squeeze(1)
        B, C, H, W = obs_image.shape

        # === 正样本 ===
        positive = self.normalizer['action'].normalize(batch['action'])
        E_pos = self.model(obs_image, positive)  # (B, 1)

        negatives_far = self.stochastic_optimizer.negative_sample_delta(B, self.model, 16)

        negatives_init = self.stochastic_optimizer.negative_sample_delta(B, self.model, -16)
        with torch.enable_grad():
            langevin_result, _ = self.stochastic_optimizer.langevin_optimize(
                energy_network=self.model,
                observations=obs_image.detach(),
                action_samples=negatives_init
            )
        negatives_near = langevin_result['action']  # Langevin 优化后的负样本
        E_neg_near = langevin_result['energies']    # 对应能量

        # === 远离正样本负样本能量 ===
        E_neg_far = self.model(obs_image, negatives_far)  # (B, n_far)

        # === 合并负样本 ===
        negatives_all = torch.cat([negatives_far, negatives_near], dim=1)  # (B, N_neg)
        E_neg_all = torch.cat([E_neg_far, E_neg_near], dim=1)  # (B, N_neg)

        # === 打乱顺序 ===
        idx = torch.randperm(E_neg_all.size(1))
        E_neg_all = E_neg_all[:, idx]
        negatives_all = negatives_all[:, idx]

        # === 计算 InfoNCE ===
        E_all = torch.cat([E_pos, E_neg_all], dim=1)  # (B, N_neg+1)
        logits = -E_all
        log_sum_exp = torch.logsumexp(logits, dim=1, keepdim=True)
        log_prob_pos = logits[:, 0:1] - log_sum_exp

        # === L2 正则 ===
        l2_reg_lambda = getattr(self, 'l2_reg_lambda', 1e-4)
        l2_reg_loss = sum(torch.norm(p, p=2) for p in self.model.parameters() if p.requires_grad)
        l2_reg_loss = l2_reg_lambda * l2_reg_loss

        loss = -log_prob_pos.mean() + l2_reg_loss
        return loss
