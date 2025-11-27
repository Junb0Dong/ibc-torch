"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from ibc.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-v', '--visualize_energy', is_flag=True, help='Generate energy distribution visualization videos')
def main(checkpoint, output_dir, device, visualize_energy):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # 兼容性加载：处理模型结构不匹配问题
    try:
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        
    except RuntimeError as e:
        if "size mismatch" in str(e) or "Error(s) in loading state_dict" in str(e):
            print(f"检测到模型结构不匹配: {e}")
            print("尝试使用兼容性加载...")
            
            # 手动创建策略实例以处理维度不匹配
            from ibc.compatibility_loader import CompatibleEbmPolicy
            
            # 从配置创建策略
            policy_cfg = cfg.policy
            policy_state_dict = payload['state_dicts']['model']
            
            # 创建兼容策略
            policy = CompatibleEbmPolicy(**policy_cfg)
            policy.load_state_dict(policy_state_dict, strict=False)
            
            # 尝试从数据集获取normalizer
            try:
                dataset = hydra.utils.instantiate(cfg.task.dataset)
                normalizer = dataset.get_normalizer()
                policy.set_normalizer(normalizer)
                print("Normalizer loaded from dataset")
            except Exception as norm_e:
                print(f"Could not load normalizer from dataset: {norm_e}")
        else:
            raise e
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)

    # 如果需要可视化能量分布
    if visualize_energy:
        from ibc.visualize_energy import EnergyVisualizer
        import numpy as np
        from ibc.env.pusht.pusht_image_env import PushTImageEnv
        
        print("开始生成能量分布可视化...")
        
        # 创建能量可视化器
        visualizer = EnergyVisualizer(policy, device=device)
        
        # 创建测试环境
        test_env = PushTImageEnv(render_size=96)
        
        # 生成几个不同状态的能量分布视频
        for i in range(2):  # 减少生成数量以节省时间
            # 重置环境到随机状态
            obs = test_env.reset()
            
            # 准备观察字典
            obs_dict = {
                'image': torch.from_numpy(obs['image']).unsqueeze(0).float().to(device),
                'agent_pos': torch.from_numpy(obs['agent_pos']).unsqueeze(0).float().to(device)
            }
            
            # 生成能量分布视频
            video_path = os.path.join(output_dir, f'energy_distribution_{i}.mp4')
            visualizer.visualize_energy_distribution(
                obs_dict=obs_dict,
                output_path=video_path,
                resolution=1024  # 降低分辨率以加快计算
            )
            
            runner_log[f'energy_video_{i}'] = wandb.Video(video_path)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
