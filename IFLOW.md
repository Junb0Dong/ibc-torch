# iFlow Context File for ibc-torch

## 项目概述

这是一个基于 PyTorch 实现的能量模型（Energy-Based Model, EBM）机器人控制项目，专注于复现 IBC（Implicit Behavior Cloning）算法，特别是针对 2D PushT 任务。项目利用能量模型来学习机器人策略，通过能量最小化原则进行动作预测。

项目基于 IBC-torch 库，结合了 Diffusion Policy (DP) 的数据和环境处理方式，在能量模型上实现 2D PushT 的实验。主要区别是用能量模型替代扩散模型，训练通过 InfoNCE 损失函数进行。

## 核心组件

### 模型架构
- `ibc.models.EBMConvMLP`: 核心能量模型，结合 CNN 特征提取和 MLP 能量计算
- `ibc.policy.ebm_image_policy.EbmUnetHybridImagePolicy`: 主策略类，实现 `compute_loss` 和 `predict_action` 方法

### 训练流程
- `ibc.workspace.simulation_pushing_pixel_workspace.SimulationPushingWorkspace`: 管理整个训练流程
- 使用 InfoNCE 损失函数进行训练
- 支持负采样和推理时的优化

### 优化器
- `ibc.optimizers.DerivativeFreeOptimizer`: 无导数优化器，用于推理时的能量最小化
- `ibc.optimizers.LangevinMCMCOptimizer`: Langevin MCMC 优化器，提供替代方案

### 数据处理
- `dataset.pusht_image_dataset.PushTImageDataset`: PushT 图像数据集
- 使用归一化处理图像、动作和代理位置数据

## 配置文件

- `ibc/config/image_pusht_diffusion_policy_cnn.yaml`: 主要配置文件，定义了网络结构、训练参数、数据加载器等

## 训练与评估

### 训练命令
```bash
python train_pusht.py --config-name=image_pusht_diffusion_policy_cnn
```

### 评估命令
```bash
python eval.py --checkpoint [checkpoint_path] -o [output_dir]
```

## 关键特性

1. **能量模型架构**: 使用 CNN+MLP 结构实现能量函数，用于评估动作-观测对的能量
2. **负采样策略**: 在训练时使用负采样技术，通过对比学习训练能量模型
3. **归一化处理**: 对动作、图像和代理位置数据进行归一化，确保数值稳定性
4. **多模态推理**: 使用优化算法（如 Derivative-Free 或 Langevin MCMC）在推理时找到低能量动作
5. **时序处理**: 支持多时间步的观测和动作序列处理

## 开发进度与挑战

根据 README 记录，项目经历了多个开发阶段，解决了以下关键问题：
- 数据归一化问题（训练数据和负样本的归一化一致性）
- 推理时动作预测的维度问题
- 时序信息在 CNN+MLP 架构中的处理
- 损失函数设计和优化

## 文件结构

- `ibc/`: 核心算法和模型代码
- `dataset/`: 数据集处理代码
- `data/`: 存放原始数据
- `experiments/`: 存放实验结果和检查点
- `notebook/`: Jupyter 笔记本文件（1D/2D 实验）
- `config/`: 配置文件

## 技术栈

- Python 3.8
- PyTorch
- Hydra (配置管理)
- OmegaConf
- WandB (日志记录)
- NumPy
- OpenCV (cv2)

## 开发约定

1. **数据格式**: PushT 数据包含图像 (`['obs']['image']`)、代理位置 (`['obs']['agent_pos']`) 和动作 (`['action']`)
2. **归一化**: 确保训练数据、负样本和推理输入的归一化一致性
3. **模型设计**: CNN+MLP 架构下实现 Markov 决策过程
4. **负采样**: 在目标边界范围内生成负样本