"""
兼容性加载器，用于加载不同版本的模型权重
"""
import torch
import torch.nn as nn
from ibc.policy.ebm_image_policy import EbmUnetHybridImagePolicy


class CompatibleEbmPolicy(EbmUnetHybridImagePolicy):
    """
    兼容性策略类，用于加载不同维度的checkpoint
    """
    def __init__(self, *args, **kwargs):
        # 移除hydra特定的参数
        kwargs = kwargs.copy()
        kwargs.pop('_target_', None)
        
        # 检查input_dim是否为514（从checkpoint中得知）
        if 'input_dim' in kwargs and kwargs['input_dim'] == 514:
            # 如果是514维度，说明是旧版本，直接使用
            super().__init__(*args, **kwargs)
        else:
            # 否则，修改参数以匹配checkpoint
            kwargs['input_dim'] = 514  # 强制使用checkpoint中的维度
            super().__init__(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        """
        重写load_state_dict方法以处理维度不匹配的参数
        """
        # 获取当前模型的state_dict
        current_state = self.state_dict()
        
        # 创建一个新的state_dict，只包含兼容的参数
        adapted_state_dict = {}
        
        for name, param in state_dict.items():
            if name in current_state:
                if param.shape == current_state[name].shape:
                    # 形状匹配，直接使用
                    adapted_state_dict[name] = param
                elif strict:
                    print(f"Warning: Shape mismatch for {name}: checkpoint {param.shape} vs current {current_state[name].shape}")
                else:
                    # 如果非严格模式，尝试填充或裁剪
                    current_param = current_state[name]
                    if len(param.shape) == len(current_param.shape):
                        # 如果维度数相同，尝试适配
                        adapted_param = self._adapt_parameter(param, current_param)
                        adapted_state_dict[name] = adapted_param
                    else:
                        # 维度数不同，跳过
                        print(f"Skipping {name} due to incompatible dimensions")
            else:
                if strict:
                    print(f"Warning: Unexpected key {name} in state_dict")
        
        # 调用父类的load_state_dict
        return super().load_state_dict(adapted_state_dict, strict=False)
    
    def _adapt_parameter(self, checkpoint_param, current_param):
        """
        适配参数张量的大小以匹配当前模型
        """
        # 创建一个新的张量，与当前参数形状相同
        adapted_param = current_param.clone()
        
        # 获取最小的尺寸
        min_sizes = [min(cp, cc) for cp, cc in zip(checkpoint_param.shape, current_param.shape)]
        
        # 复制匹配的部分
        slices = tuple(slice(0, size) for size in min_sizes)
        adapted_param[slices] = checkpoint_param[slices]
        
        return adapted_param


def load_compatible_policy(policy_config, checkpoint_state_dict):
    """
    加载兼容的策略
    """
    # 尝试创建兼容版本的策略
    try:
        policy = CompatibleEbmPolicy(**policy_config)
        policy.load_state_dict(checkpoint_state_dict, strict=False)
        return policy
    except Exception as e:
        print(f"Failed to load with CompatibleEbmPolicy: {e}")
        # 作为备选方案，尝试修改配置以匹配checkpoint
        modified_config = policy_config.copy()
        # 根据错误信息，checkpoint中的MLP输入维度是514
        if 'mlp_config' in modified_config:
            modified_config['mlp_config']['input_dim'] = 514
        elif 'input_dim' in modified_config:
            modified_config['input_dim'] = 514
        
        policy = EbmUnetHybridImagePolicy(**modified_config)
        policy.load_state_dict(checkpoint_state_dict, strict=False)
        return policy