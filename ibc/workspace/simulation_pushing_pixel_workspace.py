if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent) #  获取当前文件的父目录的父目录的父目录的路径
    sys.path.append(ROOT_DIR) #  将该路径添加到系统路径中
    os.chdir(ROOT_DIR) #  将当前工作目录更改为该路径

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from ibc.workspace.base_workspace import BaseWorkspace
from ibc.policy.simulation_pushing_pixel_policy import SimulationPushingPixelPolicy   # TODO：换成能量模型的训练和推理
from dataset.base_dataset import BaseImageDataset
from ibc.env_runner.base_image_runner import BaseImageRunner
from ibc.common.checkpoint_util import TopKCheckpointManager
from ibc.common.json_logger import JsonLogger
from ibc.common.pytorch_util import dict_apply, optimizer_to
# from ibc.model.diffusion.ema_model import EMAModel # TODO：先不用ema，后续再加，先简单的跑一遍EBM
from ibc.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class SimulationPushingWorkspace(BaseWorkspace): # 从基类BaseWorkspace继承
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)    # inherit from father class __init__ method

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: SimulationPushingPixelPolicy = hydra.utils.instantiate(cfg.policy)    # create instance of policy class

        # self.ema_model: DiffusionUnetHybridImagePolicy = None
        # if cfg.training.use_ema:
        #     self.ema_model = copy.deepcopy(self.model)  # 加载ema模型

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        

    def run(self): 
        cfg = copy.deepcopy(self.cfg)
        n_obs_steps = cfg.n_obs_steps

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        print("type of normalizer:", type(normalizer))
        
        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        # if cfg.training.use_ema:
        #     self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=( # 计算训练步骤数，即训练数据集的长度乘以训练的轮数，再除以梯度累积的次数
                len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,  # // 整数除法运算符
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        # ema: EMAModel = None
        # if cfg.training.use_ema:
        #     ema = hydra.utils.instantiate(
        #         cfg.ema,
        #         model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        # debug 模式
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger: # 使用JsonLogger类，将日志写入log_path文件
            for local_epoch_idx in range(cfg.training.num_epochs): # 训练轮数
                progress = local_epoch_idx / cfg.training.num_epochs
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):  # 训练数据循环
                        # device transfer
                        # 提取2个时间步的观测并拼接通道

                        # assert obs_dict.shape[1] >= n_obs_steps, f"序列长度{T}需≥{n_obs_steps}"
                        
                        # # 方案A：取前2个时间步（简单直观，适合离线数据集）
                        # obs_2steps = obs_dict[:, :n_obs_steps]  # 形状：(B, 2, 3, H, W)
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_actions = batch['action']
                        
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.InfoNCE_loss(batch, progress)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        # if cfg.training.use_ema:
                        #     ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0] # 当前学习率
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break # 如果达到最大训练步数，则退出循环

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)  
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                # if cfg.training.use_ema:
                #     policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log) # provision mean_score

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):  # batch from val dataloader
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.InfoNCE_loss(batch, progress)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action'][:,-1,...] # ground_truth action
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint 保存，得先rollout获得mean_score后才能保存，因此要和rollout_every保持一致
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    try:
                        # 先保存常规检查点
                        if cfg.checkpoint.save_last_ckpt:
                            last_ckpt_path = self.save_checkpoint(use_thread=False)  # 禁用线程
                            if last_ckpt_path is None:
                                print(f"Failed to save checkpoint at epoch {self.epoch}")
                        
                        # 再保存快照
                        if cfg.checkpoint.save_last_snapshot:
                            snapshot_path = self.save_snapshot(use_thread=False)  # 禁用线程
                            if snapshot_path is None:
                                print(f"Failed to save snapshot at epoch {self.epoch}")

                        # 处理指标并保存topk检查点
                        metric_dict = {key.replace('/', '_'): value for key, value in step_log.items()}
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        
                        if topk_ckpt_path is not None:
                            topk_path = self.save_checkpoint(path=topk_ckpt_path, use_thread=False)
                            if topk_path is None:
                                print(f"Failed to save topk checkpoint at epoch {self.epoch}")
                                
                    except Exception as e:
                        print(f"Checkpoint saving failed at epoch {self.epoch}: {e}")
                # ========= eval end for this epoch ==========
                policy.train()  #设置policy为训练模式

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = SimulationPushingWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
