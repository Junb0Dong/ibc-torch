#!/usr/bin/env python3
"""
能量场演化可视化脚本
"""
import sys
import os
sys.path.insert(0, '.')

import torch
import dill
import hydra
import zarr
import numpy as np
from ibc.compatibility_loader import CompatibleEbmPolicy
from ibc.visualize_energy import EnergyVisualizer
# No longer need PushTImageEnv since loading from dataset


def create_energy_evolution_video():
    print("开始生成能量场演化视频...")
    
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
    
    # 加载测试数据集（如果可用，否则使用训练集）
    try:
        test_dataset_cfg = cfg.task.get('test_dataset', cfg.task.dataset)
        test_dataset = hydra.utils.instantiate(test_dataset_cfg)
        print("Test dataset loaded")
    except Exception as e:
        test_dataset = dataset
        print(f"Using train dataset: {e}")
    
    # 直接从zarr store加载数据
    path = '/home/ps/ibc-torch/data/pusht/pusht_cchi_v7_replay.zarr'  # 假设cfg中包含path；如果不是，使用test_dataset.path如果可用
    try:
        store = zarr.open(path, mode='r')
        print("Loaded zarr store from dataset path")
    except AttributeError:
        # 如果cfg.task.dataset没有path，尝试从dataset实例获取
        try:
            path = test_dataset.path
            store = zarr.open(path, mode='r')
            print("Loaded zarr store from dataset.path")
        except AttributeError:
            raise ValueError("Could not find dataset path in cfg or dataset instance. Please provide the path manually.")
    
    episode_ends = store['meta']['episode_ends'][:]
    print(f"Episode ends loaded, total steps: {len(episode_ends)}")
    
    # 计算episode starts
    episode_start_indices = [0]
    for i in range(len(episode_ends) - 1):
        if episode_ends[i]:
            episode_start_indices.append(i + 1)
    episode_start_indices = np.array(episode_start_indices)
    num_episodes = len(episode_start_indices) - 1 if len(episode_start_indices) > 1 else 1
    print(f"Found {num_episodes} episodes")
    
    # 随机选择 episodes
    num_episodes_to_sample = 100
    steps_per_episode = 10
    selected_ep_indices = np.random.choice(num_episodes, min(num_episodes_to_sample, num_episodes), replace=False)
    
    obs_sequence = []
    
    for ep_idx in selected_ep_indices:
        start_idx = episode_start_indices[ep_idx]
        end_idx = episode_start_indices[ep_idx + 1] if ep_idx + 1 < len(episode_start_indices) else len(episode_ends)
        ep_length = end_idx - start_idx
        num_steps = min(steps_per_episode, ep_length)
        
        # 加载该episode的目标位置（假设keypoint顺序：agent, block, target）
        try:
            ep_keypoint_start = store['data']['keypoint'][start_idx].astype(np.float32)
            target_pos = ep_keypoint_start[2]  # target_pos (x,y)
            print(f"Episode {ep_idx}: target_pos = {target_pos}")
        except Exception as e:
            print(f"Could not load target_pos for ep {ep_idx}: {e}, using fixed [256,256]")
            target_pos = np.array([256.0, 256.0])
        
        for t in range(num_steps):
            global_idx = start_idx + t
            
            # 从 store 加载观测
            img = store['data']['img'][global_idx]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
            
            state = store['data']['state'][global_idx].astype(np.float32)  # 假设 (5,) : [agent_x, agent_y, block_x, block_y, block_theta]
            
            # 提取位置（世界坐标 [0,512]）
            agent_pos = state[0:2]
            block_pos = state[2:4]  # T块位置 (忽略 theta)
            
            obs_dict = {
                'image': torch.from_numpy(img).unsqueeze(0).float().to(device),  # (1, H, W, C)
                'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).float().to(device),
                # 添加用于标记的位置信息（假设 visualizer 会使用这些键来绘制）
                'block_pos': torch.from_numpy(block_pos).unsqueeze(0).float().to(device),
                'target_pos': torch.from_numpy(target_pos).unsqueeze(0).float().to(device),
            }
            # 转置图像到 channels-first (B, C, H, W)，因为模型期望此格式
            if obs_dict['image'].shape[-1] == 3:  # 如果是 (B, H, W, 3)
                obs_dict['image'] = obs_dict['image'].permute(0, 3, 1, 2)
            obs_sequence.append((obs_dict, None))  # None for action, as we're visualizing energy at observations
    
    print(f"从数据集中加载了 {len(obs_sequence)} 个观测状态（分辨率符合数据集图像大小 96x96）")
    
    # 创建可视化器
    visualizer = EnergyVisualizer(policy, device=device)
    
    # 创建输出目录
    output_dir = 'outputs/visualize_energy'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成能量场演化视频
    # 注意：分辨率已设置为 1024 以提高输出视频质量；图像分辨率来自数据集（96x96，会被上采样）
    print("开始生成能量场演化视频...")
    visualizer.visualize_energy_evolution(
        obs_sequence=obs_sequence,
        output_path=os.path.join(output_dir, 'energy_field_evolution.mp4'),
        resolution=1024,  # 输出视频分辨率，提高以符合要求
        n_frames=min(50, len(obs_sequence))  # 限制帧数
    )
    
    print("能量场演化视频生成完成！")
    
    # 注意：要在每一帧中标记 T 块、目标区域和当前 agent 位置，需要修改 EnergyVisualizer 中的渲染逻辑，
    # 使用 obs_dict 中的 'block_pos'、'target_pos' 和 'agent_pos' 来在图像上绘制圆圈或矩形等标记。
    # 例如，在 visualizer 的帧生成代码中添加 overlay 使用 cv2 或 matplotlib 绘制这些位置（需缩放到像素坐标：pos / 512 * image_size）。
    # 对于目标区域，通常是64x64方块，中心为target_pos。
    # 如果keypoint顺序不对，调整索引（例如打印keypoint_sample查看）


if __name__ == "__main__":
    create_energy_evolution_video()