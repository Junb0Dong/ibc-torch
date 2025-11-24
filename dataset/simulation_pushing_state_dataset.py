from typing import Dict
import torch
import numpy as np
import copy

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from ibc.common.pytorch_util import dict_apply
from ibc.common.replay_buffer import ReplayBuffer
from ibc.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from ibc.model.common.normalizer import LinearNormalizer
from dataset.base_dataset import BaseImageDataset
from ibc.common.normalize_util import get_image_range_normalizer

class SimulationPushingStateDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,   # 序列开始前填充的帧数
            pad_after=0,    # 序列结束后填充的帧数
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['block_translation', 'block_orientation', 'effector_target_translation', 'effector_translation',
                              'target_orientation', 'target_translation', 'actions']) # 加载image和state-action pair数据
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)  # 验证集掩码
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)  # 对训练集进行下采样

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)    # 创建一个序列采样器，用于从回放缓冲区中采样序列数据
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )   # 创建一个验证集采样器
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """
        为state-action pair数据和图像数据创建一个归一化器。
        参数 mode 指定归一化的模式，kwargs 可用于传递其他参数。
        返回一个 LinearNormalizer 实例，包含归一化器和图像数据的归一化器。
        """
        data = {
            'actions': self.replay_buffer['actions'],
            'block_translation': self.replay_buffer['block_translation'],
            'block_orientation': self.replay_buffer['block_orientation'],
            'effector_target_translation': self.replay_buffer['effector_target_translation'],
            'effector_translation': self.replay_buffer['effector_translation'],
            'target_orientation': self.replay_buffer['target_orientation'],
            'target_translation': self.replay_buffer['target_translation'],
        }   # state-action pair数据
        normalizer = LinearNormalizer() # 创建一个线性归一化器
        print("Fitting normalizer for action and agent_pos...")
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)   # 拟合归一化器，针对action和agent_pos数据
        normalizer['image'] = get_image_range_normalizer()  # 为图像数据添加归一化器
        return normalizer

    def __len__(self) -> int:
        """
        返回采样器的长度，即数据集中可采样的序列数量。
        """
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        将采样得到的数据转换为标准的数据格式。
        sample: dict, 包含所有 replay_buffer keys 的 T 长度序列
        """
        # 提取并转换为 float32
        block_translation = sample['block_translation'].astype(np.float32)
        block_orientation = sample['block_orientation'].astype(np.float32)
        effector_target_translation = sample['effector_target_translation'].astype(np.float32)
        effector_translation = sample['effector_translation'].astype(np.float32)
        target_orientation = sample['target_orientation'].astype(np.float32)
        target_translation = sample['target_translation'].astype(np.float32)
        
        actions = sample['actions'].astype(np.float32)
        

        # # 如果只需要 xy 平面，显式切片
        # agent_pos = effector_pos[:, :2]  # (T, 2)
        # block_xy  = block_pos[:, :2]     # (T, 2)，可选：作为观测的一部分

        data = {
            'obs': {
                'block_translation': block_translation, 
                'effector_translation': effector_translation,
                'effector_target_translation': effector_target_translation,
                'block_orientation': block_orientation,
                'target_orientation': target_orientation,
                'target_translation': target_translation,                
            },
            'actions': actions  # (T, action_dim)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取指定索引的样本数据，并将其转换为 PyTorch 张量。
        """
        sample = self.sampler.sample_sequence(idx)  # 从采样器中获取样本数据
        data = self._sample_to_data(sample) # 将样本数据转换为标准格式
        torch_data = dict_apply(data, torch.from_numpy) # 将数据转换为 PyTorch 张量
        return torch_data

    def get_target_bounds(self) -> np.ndarray:
        """
        获取动作数据的边界范围，用于优化器的配置。
        返回一个形状为 (2, action_dim) 的 numpy 数组，表示动作数据的最小值和最大值。
        """
        action_data = self.replay_buffer['actions']
        action_min = np.min(action_data, axis=0)  # 计算每个动作维度的最小值
        action_max = np.max(action_data, axis=0)  # 计算每个动作维度的最大值
        target_bounds = np.stack([action_min, action_max], axis=0)  # 堆叠最小值和最大值，形成边界数组
        print(f"Action target bounds: {target_bounds}")
        return target_bounds

def main():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from collections import defaultdict

    # --- 1. 加载数据集 ---
    zarr_path = os.path.expanduser('data/simulation_pushing_state.zarr')
    train_dataset = SimulationPushingStateDataset(zarr_path, val_ratio=0.1)
    val_dataset = train_dataset.get_validation_dataset()
    print("val dataset length:", len(val_dataset))
    print("val_dataset ", val_dataset)

    n_episodes_total = len(train_dataset.replay_buffer.episode_ends)
    n_train_episodes = train_dataset.train_mask.sum()
    n_val_episodes = (~train_dataset.train_mask).sum()  # 或 val_dataset.train_mask.sum()

    print("========== Dataset Split ==========")
    print(f"Total episodes: {n_episodes_total}")
    print(f"Train episodes: {n_train_episodes}")
    print(f"Val episodes:   {n_val_episodes}")
    print(f"Val ratio:      {n_val_episodes / n_episodes_total:.2%}")
    
    train_dataset.get_target_bounds()
    
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False)
    # print("Sample keys:", val_loader.keys())
    for batch in train_loader:
        print("=== Batch Structure ===")
        print("Top-level keys:", list(batch.keys()))
        print("\nObservations:")
        for k, v in batch['obs'].items():
            print("keys:",k)
            # print("value:", v)
            print(f"  {k}: {v.shape} (dtype: {v.dtype})")
        print("\nActions:", batch['actions'])
        break
    
    def compute_dataset_min_max(dataset):
        """
        遍历整个 dataset，计算每个叶子节点（如 obs/agent_pos, actions）的全局 min 和 max。
        返回格式：{ 'obs/agent_pos': {'min': ..., 'max': ...}, ... }
        """
        min_vals = defaultdict(lambda: np.inf)
        max_vals = defaultdict(lambda: -np.inf)

        def update_min_max(prefix, data):
            """递归遍历嵌套 dict，更新 min/max"""
            if isinstance(data, dict):
                for k, v in data.items():
                    new_prefix = f"{prefix}/{k}" if prefix else k
                    update_min_max(new_prefix, v)
            else:
                # 假设 data 是 array-like
                if isinstance(data, np.ndarray):
                    arr = data
                elif hasattr(data, 'numpy'):  # torch.Tensor
                    arr = data.detach().cpu().numpy()
                else:
                    arr = np.array(data)

                # 展平后更新全局 min/max
                current_min = arr.min()
                current_max = arr.max()
                if current_min < min_vals[prefix]:
                    min_vals[prefix] = current_min
                if current_max > max_vals[prefix]:
                    max_vals[prefix] = current_max

        # 遍历所有样本
        for i in range(len(dataset)):
            sample = dataset[i]
            update_min_max('', sample)

        # 合并结果
        stats = {}
        for key in min_vals.keys():
            stats[key] = {
                'min': min_vals[key],
                'max': max_vals[key]
            }
        return stats

    # 使用
    stats = compute_dataset_min_max(train_dataset)

    # 打印结果
    for key, val in stats.items():
        print(f"{key}: min={val['min']:.4f}, max={val['max']:.4f}")

if __name__ == "__main__":
    main()