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
class SimulationPushingStatePolicy(BaseImagePolicy):
    """
    能量模型policy类，主要作用：compute loss和predict_action，辅助训练
    """
    def __init__(self,
        shape_meta: dict,
        input_dim: int,
        n_action_steps: int,
        hidden_dim: int,
        hidden_depth: int,
        dropout_prob: float,
        target_bounds:  torch.Tensor,
        # stochastic_optimizer_train_samples: int,
        infer_samples: int,
        negative_samples: int,
        output_dim: int,
        # uniform_boundary_buffer: float,
        device_type: str = "cuda",
        ):
        super().__init__()
        _device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        print(f"Using device: {_device}")
        self._device = _device
        self.n_action_steps = n_action_steps
        
        action_dim = shape_meta['actions']['shape']
        
        print([v['shape'] for v in shape_meta['obs'].values()])
        print(sum(np.prod(v['shape']) for v in shape_meta['obs'].values()))
        input_dim = sum(np.prod(v['shape']) for v in shape_meta['obs'].values())
        input_dim = input_dim + np.prod(action_dim)
        print("input_dim:", input_dim)

        # 网络设置 obs -> CNN_output + action -> MLP -> action
        mlp_config = models.MLPConfig(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim,
            hidden_depth=hidden_depth, 
            dropout_prob=dropout_prob
            )

        model = models.EBMMLP(config=mlp_config)
        print("EBMsMLP model:", model)
        model.to(_device)
        self.model = model

        # TODO: 动态获取target_bounds，应该要在数据集的代码中获取，额外添加函数`get_target_bounds()`
        # 应该在workspace中获取
        print("the type of target_bounds:", type(target_bounds)) 
        target_bounds = torch.tensor(target_bounds, dtype=torch.float32)
        stochastic_optim_config = optimizers.DerivativeFreeConfig(
            bounds=target_bounds,
            infer_samples=infer_samples,
            negative_samples=negative_samples,
        )

        self.stochastic_optimizer = optimizers.DerivativeFreeOptimizer.initialize(
            stochastic_optim_config,
            device_type,
        )

        self.normalizer = LinearNormalizer()

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized_obs = {}
        for key, value in obs_dict.items():
            normalized_obs[key] = self.normalizer[key].normalize(value)

        obs_input = torch.cat([normalized_obs[k] for k in obs_dict], dim=-1)

        # # 2. 动态获取所有 key（排除图像等非状态量，如果你有）
        # # 如果全是状态量，可以直接用 list(normalized_obs.keys())
        # obs_keys = list(normalized_obs.keys())

        # # 3. 统一处理维度：假设 MLP 需要 (B, D)，所以取当前帧（如果是 3D）
        # obs_tensors = []
        # for key in obs_keys:
        #     tensor = normalized_obs[key]
        #     if tensor.dim() == 3:
        #         # (B, T, D) -> (B, D)，取最后一帧
        #         tensor = tensor[:, -1, :]
        #     elif tensor.dim() == 2:
        #         # (B, D) 保持不变
        #         pass
        #     else:
        #         raise ValueError(f"Unsupported observation shape for {key}: {tensor.shape}")
        #     obs_tensors.append(tensor)

        # 4. 拼接成 (B, D_total)
        # obs_input = torch.cat(obs_tensors, dim=-1)

        # 5. 推理
        pre_action = self.stochastic_optimizer.infer(
            obs_input.to(self._device), self.model  # 注意：你原代码写的是 input.to(...)，应该是 obs_input
        )

        # 6. 反归一化动作
        action_tensor = self.normalizer['actions'].unnormalize(pre_action)
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

        target = self.normalizer['actions'].normalize(batch['actions'])
        target = target.squeeze(1)

        negatives = self.stochastic_optimizer.negative_sample(obs_input.size(0), self.model)
        negatives = self.normalizer['actions'].normalize(negatives)

        targets = torch.cat([target.unsqueeze(dim=1), negatives], dim=1)
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(self._device)

        energy = self.model(obs_input, targets)
        logits = -1.0 * energy
        loss = F.cross_entropy(logits, ground_truth)
        return loss


    # TODO：添加last_action信息 
    def InfoNCE_loss(self, batch, progress, training=True):
        obs_agent_pos = batch['obs']['agent_pos'].squeeze(1)
        obs_image = batch['obs']['image'].squeeze(1)
        normalized_agent_pos = self.normalizer['agent_pos'].normalize(obs_agent_pos)

        B, C, H, W = obs_image.shape
        normalized_agent_pos_expanded = normalized_agent_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        input = torch.cat([obs_image, normalized_agent_pos_expanded], dim=1)

        # Positive samples: shape (B, 1, action_dim)
        positive = self.normalizer['action'].normalize(batch['action'])
        positive_pos = torch.cat([positive, normalized_agent_pos.unsqueeze(1).repeat(1, positive.size(1), 1)], dim=-1)  # 将agent_pos信息拼接到动作上
        E_pos = self.model(input, positive_pos)  # (B, 1)

        # Negative samples with curriculum learning
        # neg_samples, E_neg = self.stochastic_optimizer.negative_infer(
        #     input.to(self._device), progress, self.model
        # )
        negative = self.stochastic_optimizer.negative_sample(input.size(0), self.model) # (B, N_neg, action_dim)
        negative_pos = torch.cat([negative, normalized_agent_pos.unsqueeze(1).repeat(1, negative.size(1), 1)], dim=-1)
        E_neg = self.model(input, negative_pos)

        if E_neg.mean() < E_pos.mean():
            print(f"[Warning] E_neg ({E_neg.mean().item():.3f}) < E_pos ({E_pos.mean().item():.3f}) at progress={progress:.2f}")

        # Concatenate energies: [E_pos | E_neg]
        E_all = torch.cat([E_pos, E_neg], dim=1)  # (B, N_neg+1)

        # InfoNCE loss using stable logsumexp
        logits = -E_all  # (B, N_neg+1)
        log_sum_exp = torch.logsumexp(logits, dim=1, keepdim=True)  # (B, 1)
        log_prob_pos = logits[:, 0:1] - log_sum_exp  # (B, 1)

        # Negative log likelihood
        loss = -log_prob_pos.mean()

        return loss
