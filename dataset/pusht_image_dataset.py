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

'''
概述：PushTImageDataset 类是一个基于图像数据集的类，用于处理和采样图像数据。它继承自 BaseImageDataset 类，并使用 ReplayBuffer 来存储和访问数据。该类支持从给定的 Zarr 文件路径加载图像数据，并提供了方法来采样数据并将其转换为 PyTorch 张量。

参数：
zarr_path：字符串，Zarr 文件的路径，用于加载图像数据。
horizon：整数，默认为 1，表示采样的序列长度。
pad_before：整数，默认为 0，表示在序列开始前填充的帧数。
pad_after：整数，默认为 0，表示在序列结束后填充的帧数。
seed：整数，默认为 42，用于随机数生成器的种子。
val_ratio：浮点数，默认为 0.0，表示验证集的比例。
max_train_episodes：整数，默认为 None，表示训练集的最大集数。

返回值：
__getitem__ 方法返回一个字典，包含以下键值对：
'obs'：包含观测数据的字典，包括 'image'（图像数据，形状为 (T, 3, 96, 96)）和 'agent_pos'（代理位置数据，形状为 (T, 2)）。
'action'：动作数据，形状为 (T, 2)，并转换为 PyTorch 张量。

示例：
dataset = PushTImageDataset(zarr_path='path/to/zarr/file.zarr')
sample_data = dataset[0]  # 获取索引为 0 的样本数据
'''
class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,  # zarr 文件路径
            horizon=1,  # 采样序列长度
            pad_before=0,   # 序列开始前填充的帧数
            pad_after=0,    # 序列结束后填充的帧数
            seed=42,    # 随机种子
            val_ratio=0.0,  # 验证集比例
            max_train_episodes=None # 最大训练集集数，如果为 None 则不进行下采样
            ):
        """
        加载zarr数据，划分训练集和验证集，并创建序列采样器。
        """
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action']) # 加载image和state-action pair数据
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
        """
        创建验证集数据集，使用与训练集相同的采样器，但使用验证集掩码。
        返回一个新的 PushTImageDataset 实例，包含验证集数据。
        """
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
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
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
        参数：
        sample：一个字典，包含采样得到的图像、状态和动作数据。
        """
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255 # 图像数据的通道维度移动到第二维，并将像素值归一化到 [0, 1] 范围

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }   # data字典，包含观测数据和动作数据
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
        action_data = self.replay_buffer['action']
        action_min = np.min(action_data, axis=0)  # 计算每个动作维度的最小值
        action_max = np.max(action_data, axis=0)  # 计算每个动作维度的最大值
        target_bounds = np.stack([action_min, action_max], axis=0)  # 堆叠最小值和最大值，形成边界数组
        print(f"Action target bounds: {target_bounds}")
        return target_bounds



def main():
    import os
    zarr_path = os.path.expanduser('/home/ps/ibc-torch/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)
    # 获取一个样本（所有样本结构一致）
    sample = dataset[0]

    print(sample.keys())

    print("data['obs']:", sample['obs'])
    print("data['action']:", sample['action'])
    # 定义递归函数，收集所有键和shape
    def get_all_keys(data, parent_key=''):
        """
        递归遍历字典，收集所有键（含嵌套键）及其对应数据的shape
        data: 输入的字典（或嵌套字典）
        parent_key: 父键（用于拼接嵌套键，如'obs' + 'image' -> 'obs.image'）
        返回：字典，键为完整路径（如'obs.image'），值为对应的shape
        """
        result = {}
        # 遍历当前层级的所有键
        for key, value in data.items():
            # 拼接完整键名（父键 + 当前键，用.分隔）
            current_key = f"{parent_key}.{key}" if parent_key else key
            # 如果值是字典（含嵌套结构），递归处理
            if isinstance(value, dict):
                # 递归获取子键的信息，并合并到结果中
                result.update(get_all_keys(value, current_key))
            else:
                # 非字典值，获取其shape（假设是numpy数组或torch张量）
                try:
                    shape = value.shape
                    result[current_key] = shape # dict
                except AttributeError:
                    # 若值没有shape（如标量），记录为None或提示
                    result[current_key] = None  # 或 "scalar (no shape)"
        return result

    # 收集所有键和shape
    all_keys = get_all_keys(sample)

    # 打印结果
    print("Dataset all keys and their shapes:")
    for key, shape in all_keys.items():
        print(f"{key}: {shape}")

if __name__ == "__main__":
    main()