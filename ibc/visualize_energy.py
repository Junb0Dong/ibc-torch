"""
能量模型在2D PushT任务上的能量分布可视化模块
用于生成action空间的能量图视频，包含action和targetT块的位置及当前T块的位置
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import imageio
from ibc.env.pusht.pusht_image_env import PushTImageEnv
from ibc.policy.ebm_image_policy import EbmUnetHybridImagePolicy


class EnergyVisualizer:
    def __init__(self, policy: EbmUnetHybridImagePolicy, device='cuda'):
        self.policy = policy
        self.device = device
        self.policy.eval()
        
    def visualize_energy_distribution(self, obs_dict, output_path, target_bounds=None, 
                                    resolution=50, frame_skip=5):
        """
        可视化能量分布并生成视频
        
        Args:
            obs_dict: 观察字典，包含 'image' 和 'agent_pos'
            output_path: 输出视频路径
            target_bounds: 动作空间边界，默认为 [-1, -1], [1, 1]
            resolution: 能量图分辨率
            frame_skip: 帧间隔（用于节省计算时间）
        """
        if target_bounds is None:
            target_bounds = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32).to(self.device)
        
        # 确保图像在正确的设备上
        image = obs_dict['image'].to(self.device)
        agent_pos = obs_dict['agent_pos'].to(self.device)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (96, 96))  # 只保存能量图
        
        # 创建action网格
        x_min, y_min = target_bounds[0].cpu().numpy()
        x_max, y_max = target_bounds[1].cpu().numpy()
        
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        # 将网格点转换为动作张量
        action_grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
        action_tensor = torch.from_numpy(action_grid).float().to(self.device)
        action_tensor = action_tensor.unsqueeze(0)  # 添加batch维度
        action_tensor = action_tensor.repeat(image.shape[0], 1, 1)  # 扩展到batch大小
        
        # 计算能量值
        with torch.no_grad():
            # 获取agent_pos的归一化版本 - 这里需要根据策略的predict_action方法来处理
            obs_agent_pos = agent_pos[:, -1] if agent_pos.dim() > 2 else agent_pos
            obs_image = image
            # 处理图像维度，确保是(B, C, H, W)格式
            if obs_image.dim() == 3:  # (C, H, W) -> (1, C, H, W)
                obs_image = obs_image.unsqueeze(0)
            elif obs_image.dim() == 4 and obs_image.shape[1] == 1:  # (B, 1, C, H, W) -> (B, C, H, W) 
                obs_image = obs_image.squeeze(1)
            
            # 归一化处理
            if hasattr(self.policy, 'normalizer') and self.policy.normalizer is not None:
                try:
                    # 检查normalizer是否支持字典访问
                    if hasattr(self.policy.normalizer, '__getitem__'):
                        normalized_agent_pos = self.policy.normalizer['agent_pos'].normalize(obs_agent_pos)
                    else:
                        # 如果不支持字典访问，尝试使用其他方法
                        normalized_agent_pos = obs_agent_pos
                except (TypeError, KeyError, AttributeError):
                    # 如果访问失败，使用原始值
                    normalized_agent_pos = obs_agent_pos
            else:
                normalized_agent_pos = obs_agent_pos
            
            # 确保图像张量有正确的维度
            if len(obs_image.shape) == 3:
                B, H, W = obs_image.shape
                C = 3  # 修正图像通道数
            elif len(obs_image.shape) == 4:
                B, C, H, W = obs_image.shape
            else:
                raise ValueError(f"Unexpected image shape: {obs_image.shape}")
                
            # 如果是3维图像（即单张图片），则扩展batch维度
            if obs_image.dim() == 3:
                obs_image = obs_image.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
                B, C, H, W = obs_image.shape
                
            # 扩展agent_pos的维度以匹配图像
            normalized_agent_pos_expanded = normalized_agent_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
            model_input = torch.cat([obs_image, normalized_agent_pos_expanded], dim=1)  # 模型输入格式 (B, C+2, H, W)
            
            # 首先对动作进行归一化
            if hasattr(self.policy, 'normalizer') and self.policy.normalizer is not None:
                try:
                    # 检查normalizer是否支持字典访问
                    if hasattr(self.policy.normalizer, '__getitem__'):
                        normalized_actions = self.policy.normalizer['action'].normalize(action_tensor)
                    else:
                        # 如果不支持字典访问，尝试使用其他方法
                        normalized_actions = action_tensor
                except (TypeError, KeyError, AttributeError):
                    # 如果访问失败，使用原始值
                    normalized_actions = action_tensor
            else:
                normalized_actions = action_tensor
            
            # 将归一化的agent_pos扩展以匹配动作张量的形状
            normalized_agent_pos_for_action = normalized_agent_pos.unsqueeze(1).repeat(1, normalized_actions.shape[1], 1)
            
            # 将agent_pos与动作拼接 - 模型期望的格式
            actions_with_agent = torch.cat([normalized_actions, normalized_agent_pos_for_action], dim=-1)
            
            # 计算能量 - 使用正确格式的输入
            energies = self.policy.model(model_input, actions_with_agent)
            energies = energies.squeeze(0)  # 移除batch维度
            energy_map = energies.cpu().numpy().reshape(resolution, resolution)
            
        # 找到最小能量对应的动作
        min_energy_idx = np.unravel_index(np.argmin(energy_map), energy_map.shape)
        min_x = X[min_energy_idx]
        min_y = Y[min_energy_idx]
        
        # 获取预测动作
        try:
            with torch.no_grad():
                pred_result = self.policy.predict_action(obs_dict)
                pred_action = pred_result['action'].cpu().numpy()[0, 0, :]  # 取第一个时间步的第一个batch
        except Exception as e:
            print(f"无法获取预测动作: {e}, 使用随机动作进行演示")
            # 如果无法预测动作，则使用随机动作
            pred_action = np.random.uniform(-1, 1, size=(2,))
        
        # 将预测动作转换回原始空间（如果需要）
        if hasattr(self.policy, 'normalizer') and self.policy.normalizer is not None:
            try:
                # 检查normalizer是否支持字典访问
                if hasattr(self.policy.normalizer, '__getitem__'):
                    pred_action_unnorm = self.policy.normalizer['action'].unnormalize(
                        torch.from_numpy(pred_action).float().unsqueeze(0).unsqueeze(0)
                    ).squeeze().cpu().numpy()
                else:
                    pred_action_unnorm = pred_action
            except (TypeError, KeyError, AttributeError, RuntimeError) as e:
                # 如果访问失败，使用原始值
                pred_action_unnorm = pred_action
        else:
            pred_action_unnorm = pred_action
        
        # 生成视频帧 - 生成多帧以确保视频可播放
        for i in range(10):  # 生成10帧以确保视频时长
            # 创建能量热图
            plt.figure(figsize=(6, 6))
            plt.contourf(X, Y, energy_map, levels=50, cmap='viridis_r')
            plt.colorbar(label='Energy')
            
            # 标记最小能量点
            plt.plot(min_x, min_y, 'r*', markersize=15, label='Min Energy Action')
            
            # 标记预测动作
            plt.plot(pred_action[0], pred_action[1], 'bo', markersize=10, label='Predicted Action')
            
            # 标记目标边界
            plt.plot([x_min, x_max, x_max, x_min, x_min], 
                    [y_min, y_min, y_max, y_max, y_min], 'w--', label='Target Bounds')
            
            plt.title('Energy Distribution in Action Space')
            plt.xlabel('Action X')
            plt.ylabel('Action Y')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存能量图
            energy_fig = plt.gcf()
            energy_fig.canvas.draw()
            energy_img = np.frombuffer(energy_fig.canvas.tostring_rgb(), dtype=np.uint8)
            energy_img = energy_img.reshape(energy_fig.canvas.get_width_height()[::-1] + (3,))
            energy_img = cv2.resize(energy_img, (96, 96))  # 调整大小
            
            plt.close()
            
            # 写入视频帧（复制相同帧多次以确保视频时长）
            video_writer.write(energy_img)
        
        video_writer.release()
        print(f"能量分布视频已保存到: {output_path}")
    
    def visualize_energy_evolution(self, obs_sequence, output_path, target_bounds=None, 
                                 resolution=30, n_frames=50):
        """
        可视化能量场随时间的演化
        
        Args:
            obs_sequence: 观察序列，每个元素为(obs_dict, action)的元组
            output_path: 输出视频路径
            target_bounds: 动作空间边界，默认为 [-1, -1], [1, 1]
            resolution: 能量图分辨率
            n_frames: 生成的帧数
        """
        if target_bounds is None:
            target_bounds = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32).to(self.device)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (96, 96))
        
        # 创建action网格
        x_min, y_min = target_bounds[0].cpu().numpy()
        x_max, y_max = target_bounds[1].cpu().numpy()
        
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        print(f"开始生成能量场演化视频，总帧数: {n_frames}")

        # 如果obs_sequence没有指定，生成一些示例
        if not obs_sequence:
            # 使用当前策略生成一些示例观察
            from ibc.env.pusht.pusht_image_env import PushTImageEnv
            env = PushTImageEnv(render_size=96)
            obs_sequence = []
            obs = env.reset()
            for i in range(min(n_frames, 50)):  # 最多50个观测点
                obs_dict = {
                    'image': torch.from_numpy(obs['image']).unsqueeze(0).float().to(self.device),
                    'agent_pos': torch.from_numpy(obs['agent_pos']).unsqueeze(0).float().to(self.device)
                }
                obs_sequence.append((obs_dict, None))
                
                # 执行动（随机或使用策略）
                try:
                    with torch.no_grad():
                        pred_result = self.policy.predict_action(obs_dict)
                        action = pred_result['action'].cpu().numpy()[0, 0, :]
                    obs, _, _, _ = env.step(action * 512)  # 转换到环境动作空间
                except:
                    # 如果策略预测失败，使用随机行动
                    action = np.random.uniform(0, 512, size=(2,))
                    obs, _, _, _ = env.step(action)

        # 遫生成能量场演化视频
        for i, (obs_dict, _) in enumerate(obs_sequence[:n_frames]):
            print(f"处理第 {i+1}/{min(len(obs_sequence), n_frames)} 帧")
            
            # 确保图像在正确的设备上
            image = obs_dict['image'].to(self.device)
            agent_pos = obs_dict['agent_pos'].to(self.device)
            
            # 保持批次维度
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
            
            # 将网格点转换为动作张量
            action_grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
            action_tensor = torch.from_numpy(action_grid).float().to(self.device)
            action_tensor = action_tensor.unsqueeze(0)  # 添加batch维度
            action_tensor = action_tensor.repeat(image.shape[0], 1, 1)  # 扩展到batch大小
            
            with torch.no_grad():
                # 获取agent_pos的归一化版本
                obs_agent_pos = agent_pos
                obs_image = image
                
                # 归一化处理
                if hasattr(self.policy, 'normalizer') and self.policy.normalizer is not None:
                    try:
                        if hasattr(self.policy.normalizer, '__getitem__'):
                            normalized_agent_pos = self.policy.normalizer['agent_pos'].normalize(obs_agent_pos)
                        else:
                            normalized_agent_pos = obs_agent_pos
                    except (TypeError, KeyError, AttributeError):
                        normalized_agent_pos = obs_agent_pos
                else:
                    normalized_agent_pos = obs_agent_pos
                
                # 确保图像张量有正确的维度
                if len(obs_image.shape) == 3:
                    B, H, W = obs_image.shape
                    C = 3
                elif len(obs_image.shape) == 4:
                    B, C, H, W = obs_image.shape
                else:
                    raise ValueError(f"Unexpected image shape: {obs_image.shape}")
                    
                if obs_image.dim() == 3:
                    obs_image = obs_image.unsqueeze(0)
                    B, C, H, W = obs_image.shape
                
                # 扩展agent_pos的维度以匹配图像
                normalized_agent_pos_expanded = normalized_agent_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
                model_input = torch.cat([obs_image, normalized_agent_pos_expanded], dim=1)  # 模型输入格式 (B, C+2, H, W)
                
                # 首先对动作进行归一化
                if hasattr(self.policy, 'normalizer') and self.policy.normalizer is not None:
                    try:
                        if hasattr(self.policy.normalizer, '__getitem__'):
                            normalized_actions = self.policy.normalizer['action'].normalize(action_tensor)
                        else:
                            normalized_actions = action_tensor
                    except (TypeError, KeyError, AttributeError):
                        normalized_actions = action_tensor
                else:
                    normalized_actions = action_tensor
                
                # 将归一化的agent_pos扩展以匹配动作张量的形状
                normalized_agent_pos_for_action = normalized_agent_pos.unsqueeze(1).repeat(1, normalized_actions.shape[1], 1)
                
                # 将agent_pos与动作拼接 - 模型期望的格式
                actions_with_agent = torch.cat([normalized_actions, normalized_agent_pos_for_action], dim=-1)
                
                # 计算能量 - 使用正确格式的输入
                energies = self.policy.model(model_input, actions_with_agent)
                energies = energies.squeeze(0)  # 移除batch维度
                energy_map = energies.cpu().numpy().reshape(resolution, resolution)
                
                # 获取预测动作
                try:
                    pred_result = self.policy.predict_action(obs_dict)
                    pred_action = pred_result['action'].cpu().numpy()[0, 0, :]  # 取第一个时间步的第一个batch
                except Exception as e:
                    # 如果无法预测动作，使用随机动作
                    pred_action = np.random.uniform(-1, 1, size=(2,))
                
            # 创建能量热图
            plt.figure(figsize=(96, 96))
            plt.contourf(X, Y, energy_map, levels=50, cmap='viridis_r')
            plt.colorbar(label='Energy')
            
            # 标记预测动作
            plt.plot(pred_action[0], pred_action[1], 'bo', markersize=10, label='Predicted Action')
            
            # 标记目标边界
            plt.plot([x_min, x_max, x_max, x_min, x_min], 
                    [y_min, y_min, y_max, y_max, y_min], 'w--', label='Target Bounds')
            
            plt.title(f'Energy Field Evolution - Step {i+1}')
            plt.xlabel('Action X')
            plt.ylabel('Action Y')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存能量图
            energy_fig = plt.gcf()
            energy_fig.canvas.draw()
            energy_img = np.frombuffer(energy_fig.canvas.tostring_rgb(), dtype=np.uint8)
            energy_img = energy_img.reshape(energy_fig.canvas.get_width_height()[::-1] + (3,))
            energy_img = cv2.resize(energy_img, (96, 96))  # 调整大小
            
            plt.close()
            
            # 写入视频帧
            video_writer.write(energy_img)
        
        video_writer.release()
        print(f"能量场演化视频已保存到: {output_path}")
        
    def render_env_with_positions(self, env, agent_pos):
        """渲染带位置信息的环境图像"""
        # 使用当前环境状态，仅在图像上叠加代理位置
        img = env.render('rgb_array')
        
        # 转换为BGR用于OpenCV操作
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 在图像上标记代理位置 (agent_pos 在 [0, 512] 范围内)
        # 将世界坐标转换为图像坐标
        if len(agent_pos) >= 2:
            agent_x, agent_y = agent_pos[0], agent_pos[1]
            # 确保坐标在有效范围内
            agent_x = max(0, min(511, agent_x))
            agent_y = max(0, min(511, agent_y))
            # 转换为96x96图像坐标
            agent_pixel = (int(agent_x / 512 * 96), int(agent_y / 512 * 96))
            # 确保像素坐标在图像范围内
            agent_pixel = (min(95, max(0, agent_pixel[0])), min(95, max(0, agent_pixel[1])))
            cv2.circle(img_bgr, agent_pixel, 2, (0, 0, 255), -1)  # 红色圆点表示代理
        
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def create_energy_video_for_episode(self, policy, env, output_path, 
                                      target_bounds=None, resolution=30):
        """
        为整个episode创建能量分布视频
        """
        if target_bounds is None:
            target_bounds = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 5.0, (96*2, 96))
        
        # 重置环境
        obs = env.reset()
        obs_dict = {
            'image': torch.from_numpy(obs['image']).unsqueeze(0).float().to(self.device),
            'agent_pos': torch.from_numpy(obs['agent_pos']).unsqueeze(0).float().to(self.device)
        }
        
        step = 0
        max_steps = 100  # 限制步数
        
        while step < max_steps:
            # 获取当前状态
            current_agent_pos = obs['agent_pos']
            
            # 可视化当前步的能量分布
            self._add_energy_frame_to_video(video_writer, obs_dict, policy, 
                                          current_agent_pos, target_bounds, resolution)
            
            # 获取策略预测的动作
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
                action = action_dict['action'].cpu().numpy()[0, 0, :]  # 取第一个动作
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            if done:
                break
                
            # 更新obs_dict
            obs_dict = {
                'image': torch.from_numpy(obs['image']).unsqueeze(0).float().to(self.device),
                'agent_pos': torch.from_numpy(obs['agent_pos']).unsqueeze(0).float().to(self.device)
            }
            
            step += 1
            
        video_writer.release()
        print(f"Episode能量分布视频已保存到: {output_path}")
        
    def _add_energy_frame_to_video(self, video_writer, obs_dict, policy, agent_pos, 
                                 target_bounds, resolution):
        """添加单个能量分布帧到视频"""
        # 创建action网格
        x_min, y_min = target_bounds[0].cpu().numpy()
        x_max, y_max = target_bounds[1].cpu().numpy()
        
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        # 将网格点转换为动作张量
        action_grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
        action_tensor = torch.from_numpy(action_grid).float().to(self.device)
        action_tensor = action_tensor.unsqueeze(0)  # 添加batch维度
        batch_size = obs_dict['image'].shape[0]
        action_tensor = action_tensor.repeat(batch_size, 1, 1)  # 扩展到batch大小
        
        # 计算能量值
        with torch.no_grad():
            # 获取agent_pos的归一化版本 - 这里需要根据策略的predict_action方法来处理
            image = obs_dict['image'].to(self.device)
            agent_pos_tensor = obs_dict['agent_pos'].to(self.device)
            
            obs_agent_pos = agent_pos_tensor[:, -1] if agent_pos_tensor.dim() > 2 else agent_pos_tensor
            obs_image = image
            # 处理图像维度，确保是(B, C, H, W)格式
            if obs_image.dim() == 3:  # (C, H, W) -> (1, C, H, W)
                obs_image = obs_image.unsqueeze(0)
            elif obs_image.dim() == 4 and obs_image.shape[1] == 1:  # (B, 1, C, H, W) -> (B, C, H, W) 
                obs_image = obs_image.squeeze(1)
            
            # 归一化处理
            if hasattr(policy, 'normalizer') and policy.normalizer is not None:
                try:
                    # 检查normalizer是否支持字典访问
                    if hasattr(policy.normalizer, '__getitem__'):
                        normalized_agent_pos = policy.normalizer['agent_pos'].normalize(obs_agent_pos)
                    else:
                        # 如果不支持字典访问，尝试使用其他方法
                        normalized_agent_pos = obs_agent_pos
                except (TypeError, KeyError, AttributeError):
                    # 如果访问失败，使用原始值
                    normalized_agent_pos = obs_agent_pos
            else:
                normalized_agent_pos = obs_agent_pos
            
            # 确保图像张量有正确的维度
            if len(obs_image.shape) == 3:
                B, H, W = obs_image.shape
                C = 1  # 假设单通道，但通常图像应为3通道
            elif len(obs_image.shape) == 4:
                B, C, H, W = obs_image.shape
            else:
                raise ValueError(f"Unexpected image shape: {obs_image.shape}")
                
            # 如果是3维图像（即单张图片），则扩展batch维度
            if obs_image.dim() == 3:
                obs_image = obs_image.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
                B, C, H, W = obs_image.shape
                
            # 扩展agent_pos的维度以匹配图像
            normalized_agent_pos_expanded = normalized_agent_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
            model_input = torch.cat([obs_image, normalized_agent_pos_expanded], dim=1)  # 模型输入格式 (B, C+2, H, W)
            
            # 首先对动作进行归一化
            if hasattr(policy, 'normalizer') and policy.normalizer is not None:
                try:
                    # 检查normalizer是否支持字典访问
                    if hasattr(policy.normalizer, '__getitem__'):
                        normalized_actions = policy.normalizer['action'].normalize(action_tensor)
                    else:
                        # 如果不支持字典访问，尝试使用其他方法
                        normalized_actions = action_tensor
                except (TypeError, KeyError, AttributeError):
                    # 如果访问失败，使用原始值
                    normalized_actions = action_tensor
            else:
                normalized_actions = action_tensor
            
            # 将归一化的agent_pos扩展以匹配动作张量的形状
            normalized_agent_pos_for_action = normalized_agent_pos.unsqueeze(1).repeat(1, normalized_actions.shape[1], 1)
            
            # 将agent_pos与动作拼接 - 模型期望的格式
            actions_with_agent = torch.cat([normalized_actions, normalized_agent_pos_for_action], dim=-1)
            
            # 计算能量 - 使用正确格式的输入
            energies = policy.model(model_input, actions_with_agent)
            energies = energies.squeeze(0)  # 移除batch维度
            energy_map = energies.cpu().numpy().reshape(resolution, resolution)
            
        # 获取预测动作
        with torch.no_grad():
            pred_result = policy.predict_action(obs_dict)
            pred_action = pred_result['action'].cpu().numpy()[0, 0, :]  # 取第一个时间步的第一个batch
        
        # 创建能量热图
        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, energy_map, levels=50, cmap='viridis_r')
        plt.colorbar(label='Energy')
        
        # 标记预测动作
        plt.plot(pred_action[0], pred_action[1], 'bo', markersize=10, label='Predicted Action')
        
        # 标记当前代理位置
        plt.plot(agent_pos[0], agent_pos[1], 'ro', markersize=8, label='Current Agent Pos')
        
        # 标记目标边界
        plt.plot([x_min, x_max, x_max, x_min, x_min], 
                [y_min, y_min, y_max, y_max, y_min], 'w--', label='Target Bounds')
        
        plt.title('Energy Distribution in Action Space')
        plt.xlabel('Action X')
        plt.ylabel('Action Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存能量图
        energy_fig = plt.gcf()
        energy_fig.canvas.draw()
        energy_img = np.frombuffer(energy_fig.canvas.tostring_rgb(), dtype=np.uint8)
        energy_img = energy_img.reshape(energy_fig.canvas.get_width_height()[::-1] + (3,))
        energy_img = cv2.resize(energy_img, (96, 96))  # 调整大小以匹配环境图像
        
        plt.close()
        
        # 获取环境渲染图像
        env_img = obs_dict['image'].cpu().numpy()[0]  # 取第一个batch
        env_img = np.moveaxis(env_img, 0, -1)  # (C, H, W) -> (H, W, C)
        env_img = (env_img * 255).astype(np.uint8)  # 反归一化
        
        # 在环境图像上标记代理位置
        agent_pixel = (int(agent_pos[0] / 512 * 96), int(agent_pos[1] / 512 * 96))
        cv2.circle(env_img, agent_pixel, 2, (255, 0, 0), -1)  # 蓝色圆点表示代理
        
        # 拼接图像（能量图 + 环境图）
        combined_image = np.hstack([energy_img, env_img])
        
        # 写入视频帧
        video_writer.write(combined_image)