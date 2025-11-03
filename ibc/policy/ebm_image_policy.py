from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ibc.model.common.normalizer import LinearNormalizer
from ibc.policy.base_image_policy import BaseImagePolicy
from ibc import models, optimizers

from typing import List
class EbmUnetHybridImagePolicy(BaseImagePolicy):
    """
    能量模型policy类，主要作用：compute loss和predict_action，辅助训练
    """
    def __init__(self,
        shape_meta: dict,   # TODO：粉墨登场
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        dropout_prob: float,
        in_channels: int,
        residual_blocks: List[int],
        target_bounds:  torch.Tensor,
        stochastic_optimizer_train_samples: int,
        device_type: str = "cuda",
        ):
        super().__init__()
        _device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        print(f"Using device: {_device}")
        self._device = _device

        # shape设置
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]


        # 网络设置 obs -> CNN_output + action -> MLP -> action
        cnn_config = models.CNNConfig(in_channels, residual_blocks) # CNN配置

        # The ConvMLP's conv layer projects to 16 channels and the SpatialSoftArgmax
        # reducer returns 2 values per channel (x and y), so the reduced feature size
        # is 16 * 2 = 32. We concatenate the action_dim to this, so final mlp input
        # dim = 32 + action_dim.
        conv_out_channels = residual_blocks[-1]
        reduced_feat_dim = conv_out_channels * 2
        mlp_input_dim = reduced_feat_dim + action_dim
        # TODO：检查下卷积的输出dim和action_dim，看一下卷积的输出数据是否和action能匹配得上
        mlp_config = models.MLPConfig(
            input_dim=mlp_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
            dropout_prob=dropout_prob,
        )

        model_config = models.ConvMLPConfig(
            cnn_config=cnn_config,
            mlp_config=mlp_config,
            spatial_reduction=models.SpatialReduction.SPATIAL_SOFTMAX,
        )
        print("ConvMLP model config:", model_config)

        model = models.EBMConvMLP(config=model_config)
        print("EBMConvMLP model:", model)
        model.to(_device)
        self.model = model

        # TODO: 动态获取target_bounds，应该要在数据集的代码中获取，额外添加函数`get_target_bounds()`
        # 应该在workspace中获取
       
        print("the type of target_bounds:", type(target_bounds)) 
        target_bounds = torch.tensor(target_bounds, dtype=torch.float32)
        stochastic_optim_config = optimizers.DerivativeFreeConfig(
            bounds=target_bounds,
            train_samples=stochastic_optimizer_train_samples,
        )

        self.stochastic_optimizer = optimizers.DerivativeFreeOptimizer.initialize(
            stochastic_optim_config,
            device_type,
        )

        self.normalizer = LinearNormalizer()
        

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_agent_pos = obs_dict['agent_pos']  # (56, 2, 2)
        obs_image = obs_dict['image']          # (56, 2, 3, 96, 96)

        # # 处理时间维度：取第0个时间步（或根据需求取第1个，即[:, 1]）
        # T_idx = 0  # 选择第1个时间步（0-based索引）
        # obs_image = obs_image[:, T_idx]        # 处理后：(56, 3, 96, 96)（B, C, H, W）
        # obs_agent_pos = obs_agent_pos[:, T_idx]  # 处理后：(56, 2)（B, D）

        # 归一化处理，image不需要归一化
        normalized_agent_pos = self.normalizer['agent_pos'].normalize(obs_agent_pos)  # (56, 2)
        
        B, T, C, H, W = obs_image.shape

        normalized_agent_pos_expanded = normalized_agent_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, H, W)  # (56, 2, 96, 96)

        # 拼接输入
        input = torch.cat([obs_image, normalized_agent_pos_expanded], dim=2)  # (56, 5, 96, 96)


        # 合并B和T维度，转为4D张量（符合conv2d输入要求）
        input = input.flatten(0, 1)  # (B*T, C_total, H, W) → (56*2=112, 5, 96, 96)

        # 模型推理
        pre_action = self.stochastic_optimizer.infer(input.to(self._device), self.model)  # 此时输入为4D，正常运行

        # 拆分B*T维度回原始的B和T（如果后续需要时间维度）
        pre_action = pre_action.unflatten(0, (B, T))  # 例如(112, ...) → (56, 2, ...)

        
        # 反归一化，恢复到环境需要的原始尺度
        action_tensor = self.normalizer['action'].unnormalize(pre_action)
        

        # action_tensor_expanded = action_tensor.unsqueeze(1)

        result = {
            'action': action_tensor,
            'action_pred': action_tensor
        }
        return result
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        将传入的 normalizer 对象的状态（即其内部的参数）复制到当前策略类实例的 self.normalizer 中
        """
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # TODO: 对batch进行处理
        # print("bath size:", batch['obs']['image'].shape)
        # print("bath size:", batch['obs']['agent_pos'].shape)

        obs_agent_pos = batch['obs']['agent_pos']  # shape: (B, 2)，B为批次大小
        obs_image = batch['obs']['image']          # shape: (B, 1, 3, 96, 96)
        
        obs_image = obs_image.squeeze(1)  # 去掉时间维度，变为 (B, 3, 96, 96)
        obs_agent_pos = obs_agent_pos.squeeze(1)  # 去掉时间维度，变为 (B, 2)，不考虑时间维度，纯粹的Markov Chain

        normalized_agent_pos = self.normalizer['agent_pos'].normalize(obs_agent_pos)  # 归一化低维位置
        # normalized_image = obs_image / 255.0  # 图像通常归一化到[0,1]（若数据集已处理可省略）

        B, C, H, W = obs_image.shape
        normalized_agent_pos_expanded = normalized_agent_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # (B,2,96,96)
        input = torch.cat([obs_image, normalized_agent_pos_expanded], dim=1)  # (B, 5, 96, 96)

        target = self.normalizer['action'].normalize(batch['action']) # 对动作进行归一化处理
        target = target.squeeze(1)  # 去掉时间维度，变为 (B, action_dim)

        # Generate N negatives, one for each element in the batch: (B, N, D).
        negatives = self.stochastic_optimizer.sample(input.size(0), self.model)
        # 对动作进行归一化了后，对negatives也进行归一化
        # negatives = self.normalizer['action'].normalize(negatives)

        # Merge target and negatives: (B, N+1, D).
        targets = torch.cat([target.unsqueeze(dim=1), negatives], dim=1)

        # Generate a random permutation of the positives and negatives.
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]

        # Get the original index of the positive. This will serve as the class label
        # for the loss.
        ground_truth = (permutation == 0).nonzero()[:, 1].to(self._device)

        # For every element in the mini-batch, there is 1 positive for which the EBM
        # should output a low energy value, and N negatives for which the EBM should
        # output high energy values.
        energy = self.model(input, targets)

        # Interpreting the energy as a negative logit, we can apply a cross entropy loss
        # to train the EBM.
        logits = -1.0 * energy
        loss = F.cross_entropy(logits, ground_truth)
        return loss
    
