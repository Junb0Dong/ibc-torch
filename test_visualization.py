#!/usr/bin/env python3
"""
简化版的能量分布可视化测试脚本
"""
import sys
import os
sys.path.insert(0, '.')

import torch
import dill
import hydra
from ibc.compatibility_loader import CompatibleEbmPolicy
from ibc.visualize_energy import EnergyVisualizer
from ibc.env.pusht.pusht_image_env import PushTImageEnv
import numpy as np


def test_visualization():
    print("开始测试能量分布可视化...")
    
    # 加载checkpoint
    checkpoint_path = 'outputs/epoch=0600-test_mean_score=0.639.ckpt'
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # 创建策略
    policy_cfg = cfg.policy
    policy_state_dict = payload['state_dicts']['model']
    
    print("创建策略...")
    policy = CompatibleEbmPolicy(**policy_cfg)
    policy.load_state_dict(policy_state_dict, strict=False)
    
    # 尝试从数据集获取normalizer
    try:
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        policy.set_normalizer(normalizer)
        print("Normalizer loaded from dataset")
    except Exception as e:
        print(f"Could not load normalizer from dataset: {e}")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    policy.to(device)
    policy.eval()
    
    print(f"Policy loaded on device: {device}")
    
    # 创建可视化器
    visualizer = EnergyVisualizer(policy, device=device)
    
    # 创建一个简单的测试观察
    print("创建测试观察...")
    test_env = PushTImageEnv(render_size=96)
    obs = test_env.reset()
    
    obs_dict = {
        'image': torch.from_numpy(obs['image']).unsqueeze(0).float().to(device),
        'agent_pos': torch.from_numpy(obs['agent_pos']).unsqueeze(0).float().to(device)
    }
    
    print("开始可视化...")
    output_dir = 'outputs/visualize_energy'
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用较低的分辨率以节省计算资源
    visualizer.visualize_energy_distribution(
        obs_dict=obs_dict,
        output_path=os.path.join(output_dir, 'test_energy_simple.mp4'),
        resolution=15,  # 降低分辨率
        frame_skip=1
    )
    
    print("能量分布可视化测试完成！")


if __name__ == "__main__":
    test_visualization()