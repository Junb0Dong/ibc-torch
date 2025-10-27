# 基于ibc-torch的pushT复现
依赖ibc-torch库，结合dp对数据和环境的处理，在能量模型上实现2DpushT的实验。

都是照搬DP的代码

workspace中管理整个的训练流程，有什么问题先从workspace入手

## 文件结构
- ibc目录存放能量模型算法相关的文件
- data中存放收集的数据
- experiments中存放实验结果
- notebook中是1D/2D的notebook
- dataset中是dataloader处理数据集的代码

## 环境设置
python版本为3.8
> 有bug! T_T

**血泪教训，直接用dp的环境，自己配的环境有bug**

## 开发进度
**10.22:** 
- 将之前的预备代码准备好了：common, normalizer, model。现在进入最关键的时候，准备ebm的policy
- policy目前只包含`compute_loss`,`predict_action`两个函数，训练的主程序在workspace中实现
- 相比于dp，ebm的yaml文件还要包含网络构造的参数

**10.23：**
- debugs, targetbounds的获取以及numpy->list->toarch.tensor的转换
- pushT dataset的shape：`B, T, shape` B为batch size，T为时间步，shape为数据维度

**10.24：**
**解决训练时遇到的问题**
- [x] debug发现负采样没有进行归一化，如果归一化后，负采样的噪声是添加到归一化的数据上还是归一化前的数据 
   归一化后loss正常了，但是现在有个问题是model的输入也要是归一化的，输出是energy，那在**推理时采样也应该是归一化**的，采样后进行反归一化输出action。进行归一化的目的是将action与image数据维度同步。
- [ ] 验证集时序问题
- [x] 训练loss和验证loss都为0
   这是因为对数据集的action进行归一化了，没有对negatives进行归一化，一个数值在[-1, 1]之间一个数值在[12,511]之间，这导致交叉熵loss从而0
- [ ] 训练action_mse_loss特别大
- [x] action的xy都在左上角[12, 25]附近
   这是因为归一化没有对齐的原因，对训练数据进行归一化，模型学到的就是[-1, 1]处能量是比较低的，在sample是没有进行归一化，所以能量模型认为越靠近[-1, 1]区间的动作越好，所以就使得x,y都很小，偏向左上角

**开发时需要考虑几个问题：**
1. 数据集格式

   pushT中的数据包含
   - `['obs']['image']:tensor(B, 3, 96, 96)`, 大小在-1到1之间
   - `['obs']['agent_pos']:tensor(B, 2)`, 大小在[12, 25][511, 511]之间
   - `['action']:tensor(B, 2)`, 大小在[12, 25][511, 511]之间
2. target_bounds：这个会影响负采样的范围，是否要和训练范围一致？一致的话机械臂如何泛化？
   > 是会影响采样范围，但为了和image对应上，对于所有的数据都scale到`[0, 1]`或`[-1, 1]`之间，后买年使用的时候再`unnormalize`。所以给`target_bounds`的范围定义为`[[-1, -1], [1, 1]]`
3. 数据归一化的问题，究竟是哪些数据归一化了，是否会影响action？
   > `action`, `agent_pos`, `sample`的都要进行归一化，会影响`action`，一定要对齐归一化。sample直接定义`target_bounds`在`[-1, 1]`之间，最后再反归一化给`pred_action`
4. 解决loss和prediction的维度，loss是Markov，prediction有两个时间戳
    ```bash
    # loss
    batch agent_pos shape: torch.Size([64, 1, 2])
    batch image shape: torch.Size([64, 1, 3, 96, 96])
    # prediction
    batch agent_pos shape: torch.Size([56, 2, 2])
    batch image shape: torch.Size([56, 2, 3, 96, 96])
    ```
   > 解决办法：先都用一个时间戳的信息，把所有训练流程都简化，跑起来再说
5. 这么多bug，为什么不直接放到ibc中跑？
   > 因为需要env_runner，对数据处理，记录视频，包括后面的遥操收集数据



   