import dataclasses
import enum
from functools import partial
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import CoordConv, GlobalAvgPool2d, GlobalMaxPool2d, SpatialSoftArgmax

# only run this file need rto add `export PYTHONPATH=$PWD`

class ActivationType(enum.Enum):
    RELU = nn.ReLU
    SELU = nn.SiLU


@dataclasses.dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    hidden_depth: int
    dropout_prob: Optional[float] = None
    activation_fn: ActivationType = ActivationType.RELU


def pos_to_heatmap(agent_pos, H, W, sigma=0.1):
    """
    agent_pos: (B, 2) in [-1, 1]  -> normalized coordinates
    returns: (B, 1, H, W)  Gaussian heatmap
    """
    B = agent_pos.size(0)
    device = agent_pos.device

    # Create coordinate grid in [-1, 1]
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0)  # (1, 2, H, W)

    # Compute Gaussian centered at agent_pos
    pos = agent_pos.view(B, 2, 1, 1)
    dist = torch.sum((grid - pos) ** 2, dim=1, keepdim=True)
    heatmap = torch.exp(-dist / (2 * sigma ** 2))
    return heatmap

class MLP(nn.Module):
    """A feedforward multi-layer perceptron."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()

        dropout_layer: Callable
        if config.dropout_prob is not None:
            dropout_layer = partial(nn.Dropout, p=config.dropout_prob)
        else:
            dropout_layer = nn.Identity

        layers: Sequence[nn.Module]
        if config.hidden_depth == 0:
            layers = [nn.Linear(config.input_dim, config.output_dim)]
        else:
            layers = [
                nn.Linear(config.input_dim, config.hidden_dim),
                config.activation_fn.value(),
                dropout_layer(),
            ]
            for _ in range(config.hidden_depth - 1):
                layers += [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    config.activation_fn.value(),
                    dropout_layer(),
                ]
            layers += [nn.Linear(config.hidden_dim, config.output_dim)]
        layers = [layer for layer in layers if not isinstance(layer, nn.Identity)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class EBMMLP(nn.Module):
    """A feedforward multi-layer perceptron."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config

        dropout_layer: Callable
        if config.dropout_prob is not None:
            dropout_layer = partial(nn.Dropout, p=config.dropout_prob)
        else:
            dropout_layer = nn.Identity

        layers: Sequence[nn.Module]
        if config.hidden_depth == 0:
            layers = [nn.Linear(config.input_dim, config.output_dim)]
        else:
            layers = [
                nn.Linear(config.input_dim, config.hidden_dim),
                config.activation_fn.value(),
                dropout_layer(),
            ]
            for _ in range(config.hidden_depth - 1):
                layers += [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    config.activation_fn.value(),
                    dropout_layer(),
                ]
            layers += [nn.Linear(config.hidden_dim, config.output_dim)]
        layers = [layer for layer in layers if not isinstance(layer, nn.Identity)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # print("x.shape:", x.shape)  # 输出: [8, 1]
        # print("y.shape:", y.shape)  # 输出: [8, 16384, 1]
        fused = torch.cat([x.expand(-1, y.size(1), -1), y], dim=-1)
        # print("fused shape:", fused.shape)
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D)
        out = self.net(fused)
        out = out.view(B, N, self.config.output_dim)
        energies = out.mean(dim=-1) # 平均所有输出维度作为能量值
        return energies
    
    # def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     if self.coord_conv:
    #         x = CoordConv()(x)
    #     out = self.cnn(x, activate=True)
    #     out = F.relu(self.conv(out))
    #     out = self.reducer(out)
    #     fused = torch.cat([out.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
    #     B, N, D = fused.size()
    #     fused = fused.reshape(B * N, D)
    #     out = self.mlp(fused)
    #     return out.view(B, N)

class ResidualBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        activation_fn: ActivationType = ActivationType.RELU,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.activation = activation_fn.value()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(x)
        out = self.conv2(out)
        return out + x


@dataclasses.dataclass(frozen=True)
class CNNConfig:
    in_channels: int
    blocks: Sequence[int] = dataclasses.field(default=(16, 32, 32))
    activation_fn: ActivationType = ActivationType.RELU


class CNN(nn.Module):
    """A residual convolutional network."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()

        depth_in = config.in_channels

        layers = []
        for depth_out in config.blocks:
            layers.extend(
                [
                    nn.Conv2d(depth_in, depth_out, kernel_size=3, stride=1, padding=1),
                    ResidualBlock(depth_out, config.activation_fn),
                    nn.MaxPool2d(2),
                ]
            )
            depth_in = depth_out

        self.net = nn.Sequential(*layers)
        self.activation = config.activation_fn.value()

    def forward(self, x: torch.Tensor, activate: bool = False) -> torch.Tensor:
        out = self.net(x)
        if activate:
            return self.activation(out)
        return out


class SpatialReduction(enum.Enum):
    SPATIAL_SOFTMAX = SpatialSoftArgmax
    AVERAGE_POOL = GlobalAvgPool2d
    MAX_POOL = GlobalMaxPool2d


@dataclasses.dataclass(frozen=True)
class ConvMLPConfig:
    cnn_config: CNNConfig
    mlp_config: MLPConfig
    spatial_reduction: SpatialReduction = SpatialReduction.AVERAGE_POOL
    coord_conv: bool = False


class ConvMLP(nn.Module):
    def __init__(self, config: ConvMLPConfig) -> None:
        super().__init__()

        self.coord_conv = config.coord_conv

        self.cnn = CNN(config.cnn_config)
        self.conv = nn.Conv2d(config.cnn_config.blocks[-1], 16, 1)
        self.reducer = config.spatial_reduction.value()
        self.mlp = MLP(config.mlp_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.coord_conv:
            x = CoordConv()(x)
        out = self.cnn(x, activate=True)
        out = F.relu(self.conv(out))
        out = self.reducer(out)
        out = self.mlp(out)
        return out


class EBMConvMLP(nn.Module):
    def __init__(self, config: ConvMLPConfig) -> None:
        super().__init__()

        self.coord_conv = config.coord_conv

        self.cnn = CNN(config.cnn_config)
        # self.conv = nn.Conv2d(config.cnn_config.blocks[-1], 16, 1)
        self.reducer = config.spatial_reduction.value()
        self.mlp = MLP(config.mlp_config)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        
        obs_image = x[:, :3, :, :]  # (B, 3, 96, 96)
        obs_agent_pos = x[:, 3:5, :1, :1]   # (B, 2)
        
        heatmap = pos_to_heatmap(obs_agent_pos, H=96, W=96, sigma=0.1)
        x = torch.cat([obs_image, heatmap], dim=1)  # (B, 4, 96, 96)

        # 再送入 CNN
        # features = self.cnn(x)  # (B, C, H', W')
        # if self.coord_conv:
        #     x = CoordConv()(x)
        out = self.cnn(x, activate=True)
        # out = F.relu(self.conv(out))
        out = self.reducer(out)
        fused = torch.cat([out.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D)
        out = self.mlp(fused)
        return out.view(B, N)
    
class ConvMaxPool(nn.Module):
    """与TensorFlow版本对应的PyTorch卷积+最大池化网络"""
    def __init__(
        self, 
        in_channels: int,  # 输入通道数（无默认值，放前面）
        blocks: Sequence[int] = (16, 32, 32),  # 每一层的输出通道数（有默认值，放后面）
        activation_fn: ActivationType = ActivationType.RELU  # 激活函数类型
    ):
        super().__init__()  # 必须先调用父类初始化方法
        depth_in = in_channels
        layers = []
        
        # 将ActivationType转换为实际的PyTorch激活层
        activation_map = {
            ActivationType.RELU: nn.ReLU(inplace=True),
        }
        activation = activation_map[activation_fn]  # 获取激活层实例
        
        for depth_out in blocks:
            # 每层结构：Conv2d → 激活函数 → MaxPool2d
            layers.extend([
                nn.Conv2d(
                    in_channels=depth_in,
                    out_channels=depth_out,
                    kernel_size=3,
                    stride=1,
                    padding=1  # same padding，保持尺寸
                ),
                activation,  # 使用转换后的激活层实例
                nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2池化，步长2
            ])
            depth_in = depth_out
        
        # 补充全局平均池化层，将特征图压缩为 [batch_size, C, 1, 1]
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        # 将层序列保存为实例属性（关键：forward中需要调用）
        self.features = nn.Sequential(*layers)
        
        self.out_channels = blocks[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状：(batch_size, in_channels, height, width)（PyTorch通道在前）
        x = self.features(x)  # 通过特征提取层
        x = torch.flatten(x, 1)  # 展平为 (batch_size, 最后一层输出通道数)
        return x


def get_conv_maxpool(nchannels):
    """实例化卷积+最大池化网络（与原函数接口保持一致）"""
    return ConvMaxPool(nchannels)

def dense(in_width, out_width):
    """线性层，无激活函数（对应原TensorFlow的dense函数）"""
    layer = nn.Linear(in_width, out_width)  # 输入输出维度均为width
    # 初始化权重和偏置为正态分布（对应原代码的'normal'初始化）
    nn.init.normal_(layer.weight, mean=0.0, std=1.0)
    nn.init.normal_(layer.bias, mean=0.0, std=1.0)
    return layer


class ResNetDenseBlock(nn.Module):
    """密集残差块（对应原ResNetDenseBlock）"""
    def __init__(self, width):
        super(ResNetDenseBlock, self).__init__()
        self.dense0 = dense(width, width // 4)
        self.dense1 = dense(width // 4, width // 4)
        self.dense2 = dense(width // 4, width)
        self.dense3 = dense(width//4 , width)  # 用于维度匹配的投影层

        self.activation0 = nn.ReLU()
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ReLU()

    def forward(self, x, training=True):
        y = self.dense0(self.activation0(x))
        y = self.dense1(self.activation1(y))
        y = self.dense2(self.activation2(y))
        
        # 若输入x与输出y维度不匹配，则对x进行投影
        if x.size() != y.size():
            x = self.dense3(self.activation3(x))
        
        return x + y  # 残差连接


class DenseResnetValue(nn.Module):
    """密集残差网络值函数（对应原DenseResnetValue）"""
    def __init__(self, input_dim:int, width: int = 512, num_blocks: int = 2):
        super(DenseResnetValue, self).__init__()
        self.dense0 = dense(input_dim, width)  # 初始线性层
        # 残差块列表（使用ModuleList管理子模块）
        self.blocks = nn.ModuleList([ResNetDenseBlock(width) for _ in range(num_blocks)])
        self.dense1 = dense(width, 1)  # 最终输出层（映射到1维）

    def forward(self, x, training=True):
        # 前向传播逻辑与原call方法一致
        x = self.dense0(x)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.dense1(x)
        return x


class ConvResnetEBM(nn.Module):
    """合并卷积网络与密集残差网络的EBM模型"""
    def __init__(
        self,
        # 卷积网络配置
        cnn_in_channels: int,
        cnn_blocks: Sequence[int] = (16, 32, 32),
        cnn_activation: ActivationType = ActivationType.RELU,
        # 残差网络配置
        resnet_width: int = 512,
        resnet_num_blocks: int = 2,
        # 额外配置
        coord_conv: bool = False,
        action_dim: int = 2  # 动作维度（与观察特征融合）
    ):
        super().__init__()
        self.coord_conv = coord_conv
        self.action_dim = action_dim

        # 1. 初始化卷积特征提取网络
        self.cnn = ConvMaxPool(
            in_channels=cnn_in_channels + (2 if coord_conv else 0),  # 若使用坐标卷积则增加2个通道
            blocks=cnn_blocks,
            activation_fn=cnn_activation
        )

        # 2. 计算融合后的输入维度（CNN输出 + 动作维度）
        fusion_dim = self.cnn.out_channels + self.action_dim[0]

        # 3. 初始化密集残差值网络（处理融合特征）
        self.resnet_value = DenseResnetValue(
            input_dim=fusion_dim,
            width=resnet_width,
            num_blocks=resnet_num_blocks
        )

    # TODO：目前不用坐标卷积，代码未验证   
    def _add_coord_channels(self, x: torch.Tensor) -> torch.Tensor:
        """为输入图像添加坐标通道（-1到1范围）"""
        B, _, H, W = x.shape
        # 生成坐标网格 (H, W)
        y_coords = torch.linspace(-1.0, 1.0, H, device=x.device)
        x_coords = torch.linspace(-1.0, 1.0, W, device=x.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)

        # 扩展为 (B, 1, H, W) 并重复批次维度
        y_grid = y_grid.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 1, H, W)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 1, H, W)
        
        # 拼接坐标通道 (B, C+2, H, W)
        return torch.cat([x, y_grid, x_grid], dim=1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        前向传播：融合观察与动作，输出能量值
        Args:
            obs: 图像观察 (B, C, H, W)
            actions: 动作 (B, A) 或 (B, N, A)（支持多动作批量输入）
        Returns:
            energy: 能量值 (B,) 或 (B, N)
        """
        # 处理坐标卷积
        if self.coord_conv:
            obs = self._add_coord_channels(obs)  # (B, C+2, H, W)
        
        # 1. 提取图像特征
        obs_feat = self.cnn(obs)  # (B, C_out)
        
        # 2. 处理多动作情况（如批量评估多个动作）
        if actions.dim() == 3:
            # 动作形状: (B, N, A) → 扩展观察特征匹配维度
            B, N, A = actions.shape
            obs_feat = obs_feat.unsqueeze(1).repeat(1, N, 1)  # (B, N, C_out)
        else:
            # 动作形状: (B, A) → 保持维度一致
            obs_feat = obs_feat.unsqueeze(1)  # (B, 1, C_out)
            actions = actions.unsqueeze(1)  # (B, 1, A)
        
        # 3. 融合观察特征与动作
        fusion = torch.cat([obs_feat, actions], dim=-1)  # (B, N, C_out + A)
        B, N, D = fusion.shape
        fusion_flat = fusion.reshape(B * N, D)  # 展平为 (B*N, D) 便于MLP处理
        
        # 4. 计算能量值
        energy_flat = self.resnet_value(fusion_flat).squeeze(-1)  # (B*N,)
        energy = energy_flat.reshape(B, N)  # (B, N)
        
        # 若输入为单动作，压缩维度为 (B,)
        if N == 1:
            energy = energy.squeeze(1)
            
        return energy.view(B, N)

# before run 
# export PYTHONPATH="$PWD"
# 
if __name__ == "__main__":
    # config = ConvMLPConfig(
    #     cnn_config=CNNConfig(5),
    #     mlp_config=MLPConfig(16, 128, 2, 2),
    #     spatial_reduction=SpatialReduction.AVERAGE_POOL,
    #     coord_conv=True,
    # )

    # net = ConvMLP(config)
    # print(net)

    # x = torch.randn(2, 3, 96, 96)
    # with torch.no_grad():
    #     out = net(x)
    # print(out.shape)

    # config = MLPConfig(
    #     input_dim=2, hidden_dim=128, output_dim=1, hidden_depth=3, dropout_prob=0.1
    # )

    # net = EBMMLP(config)
    # print(net)

    # x = torch.randn(8, 1)
    # y = torch.randn(8, 1, 1)
    # with torch.no_grad():
    #     out = net(x, y)
    # print(out.shape)
    
    # # test ResnetDenseBlock
    # model = DenseResnetValue(width=512, num_blocks=1)
    # input_tensor = torch.randn(32, 512)  # 批量大小32，输入维度512
    # output = model(input_tensor, training=True)
    # print(output.shape)
    
    
    # # test ConvMaxPool    
    # model = ConvMaxPool(in_channels=3, blocks=[32,64,128,256])
    # print(model)
    # # 测试前向传播（输入形状：[batch_size=16, 3, 96, 128]）
    # input_tensor = torch.randn(16, 3, 96, 128)
    # output = model(input_tensor)
    # print(output.shape) 
    
    # test ConvResnetEBM
    model = ConvResnetEBM(
        cnn_in_channels=3, 
        cnn_blocks=[32,64,128,256], 
        resnet_width=512, 
        resnet_num_blocks=1)
    print(model)
    image = torch.randn(64, 3, 240, 320)
    actions = torch.randn(64, 2)
    
    output = model(image, actions)
    print(output.shape)