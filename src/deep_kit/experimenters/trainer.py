import os
import math
import numpy as np
from rich.progress import track

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, ConcatDataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from .operator import Operator
from ..utils import setup_logger, find_class


class MyDistributedDataParallel(DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Trainer(Operator):
    def __init__(self, cfg):
        super().__init__(cfg)

        self._init_seed()
        self._init_device()
        self._init_dataloaders()
        cls_model, self.path_file_model = find_class(cfg.model.name, 'model')
        if 'method' in cfg:
            cls_model, self.path_file_method = find_class(cfg.method.name, 'method')
        self.model = cls_model(cfg)
        if torch.__version__.startswith('2.'):
            if cfg.exp.compile_model:
                self.model = torch.compile(self.model, dynamic=True)
        self._init_dirs()
        self._init_loggers()
        self._init_writer()
        self._init_log_basic_info()

    def _init_dataloaders(self):
        cls_dataset, self.path_file_dataset = find_class(self.cfg.dataset.name, 'dataset')

        if self.cfg.exp.mode == 'train':
            if self.cfg.exp.customize_dataloader:
                self.train_set, self.train_loader = cls_dataset(mode='train', cfg=self.cfg)
                self.val_set, self.val_loader = cls_dataset(mode='val', cfg=self.cfg)
            else:
                if self.cfg.model.task_sequential:
                    self.train_sets = [cls_dataset(mode='train', cfg=self.cfg, task=task) 
                                       for task in self.cfg.dataset.train_tasks]
                    self.val_sets = [cls_dataset(mode='val', cfg=self.cfg, task=task) 
                                     for task in self.cfg.dataset.train_tasks]

                    if self.cfg.var.is_parallel:
                        self.sampler_trains = [DistributedSampler(dataset) for dataset in self.train_sets]
                        shuffle_train = False
                    else:
                        shuffle_train = not issubclass(type(self.train_sets[0]), IterableDataset)
                    
                    self.train_loaders = [
                        DataLoader(dataset=self.train_sets[i],
                                   batch_size=self.cfg.exp.train.batch_size,
                                   collate_fn=getattr(self.train_sets[i], 'get_batch', None),
                                   num_workers=self.cfg.exp.n_workers,
                                   shuffle=shuffle_train,
                                   pin_memory=True,
                                   drop_last=True,
                                   sampler=self.sampler_trains[i]
                                   if self.cfg.var.is_parallel and not issubclass(type(self.train_set), IterableDataset) else None,
                                   worker_init_fn=lambda x: np.random.seed(42 + x)
                                   )
                        for i in range(len(self.cfg.dataset.train_tasks))]
                    self.val_loaders = [
                        DataLoader(dataset=self.val_sets[i],
                                   batch_size=self.cfg.exp.val.batch_size,
                                   collate_fn=getattr(self.val_sets[i], 'get_batch', None),
                                   num_workers=self.cfg.exp.n_workers,
                                   shuffle=False,
                                   pin_memory=True,
                                   worker_init_fn=lambda x: np.random.seed(42 + x)
                                   )
                        for i in range(len(self.cfg.dataset.train_tasks))]
                    
                else:
                    self.train_set = cls_dataset(mode='train', cfg=self.cfg)
                    self.val_set = cls_dataset(mode='val', cfg=self.cfg)

                    if self.cfg.var.is_parallel:
                        self.sampler_train = DistributedSampler(self.train_set)
                        shuffle_train = False
                    else:
                        shuffle_train = not issubclass(type(self.train_set), IterableDataset)

                    self.train_loader = DataLoader(
                        dataset=self.train_set,
                        batch_size=self.cfg.exp.train.batch_size,
                        collate_fn=getattr(self.train_set, 'get_batch', None),
                        num_workers=self.cfg.exp.n_workers,
                        shuffle=shuffle_train,
                        pin_memory=True,
                        drop_last=True,
                        sampler=self.sampler_train
                        if self.cfg.var.is_parallel and not issubclass(type(self.train_set), IterableDataset) else None,
                        worker_init_fn=lambda x: np.random.seed(42 + x)
                    )
                    self.val_loader = DataLoader(
                        dataset=self.val_set,
                        batch_size=self.cfg.exp.val.batch_size,
                        collate_fn=getattr(self.val_set, 'get_batch', None),
                        num_workers=self.cfg.exp.n_workers,
                        shuffle=False,
                        pin_memory=True,
                        worker_init_fn=lambda x: np.random.seed(42 + x)
                    )
        elif self.cfg.exp.mode != 'test':
            raise ValueError

        if self.cfg.exp.customize_dataloader:
            self.test_set, self.test_loader = cls_dataset(mode='test', cfg=self.cfg)
        else:
            if self.cfg.model.task_sequential:
                self.test_sets = [cls_dataset(mode='test', cfg=self.cfg, task=task)
                                  for task in self.cfg.dataset.test_tasks]
                self.test_loaders = [
                    DataLoader(dataset=self.test_sets[i],
                               batch_size=self.cfg.exp.test.batch_size,
                               collate_fn=getattr(self.test_sets[i], 'get_batch', None),
                               num_workers=self.cfg.exp.n_workers,
                               shuffle=False,
                               pin_memory=True,
                               )
                    for i in range(len(self.cfg.dataset.test_tasks))]
            else:
                self.test_set = cls_dataset(mode='test', cfg=self.cfg)
                self.test_loader = DataLoader(
                    dataset=self.test_set,
                    batch_size=self.cfg.exp.test.batch_size,
                    collate_fn=getattr(self.test_set, 'get_batch', None),
                    num_workers=self.cfg.exp.n_workers,
                    shuffle=False,
                    pin_memory=True,
                )

    def _init_loggers(self):
        path_all = os.path.join(self.path_log, 'log_all.txt')
        path_train = os.path.join(self.path_log, 'log_train.txt')
        path_val = os.path.join(self.path_log, 'log_val.txt')
        path_checkpoints = os.path.join(self.path_log, 'log_checkpoints.txt')
        self.logger_extra = setup_logger('extra', path_all)
        self.logger_train = setup_logger('train', path_all, path_train)
        self.logger_val = setup_logger('val', path_all, path_val)
        self.logger_checkpoints = setup_logger('checkpoints', path_all, path_checkpoints)

    def _get_optimizer(self, params):
        name_opt = self.cfg.exp.train.optimizer.name
        cfg_opt = self.cfg.exp.train.optimizer[name_opt]

        if name_opt == 'sgd':
            optimizer = optim.SGD(
                params=params,
                lr=self.cfg.exp.train.optimizer.lr,
                weight_decay=cfg_opt.weight_decay,
                momentum=cfg_opt.momentum,
                nesterov=cfg_opt.nesterov,
            )
        elif name_opt == 'adam':
            optimizer = optim.Adam(
                params=params,
                lr=self.cfg.exp.train.optimizer.lr,
                weight_decay=cfg_opt.weight_decay,
            )
        elif name_opt == 'adamw':
            optimizer = optim.AdamW(
                params=params,
                lr=self.cfg.exp.train.optimizer.lr,
                weight_decay=cfg_opt.weight_decay,
            )
        else:
            raise NotImplementedError(f'Unknown optimizer: {name_opt}')

        name_sch = self.cfg.exp.train.scheduler.name
        if name_sch is not None:
            cfg_sch = self.cfg.exp.train.scheduler[name_sch]
            name_sch = name_sch.lower()
            if name_sch == 'cycliclr':
                scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=cfg_sch.lr_base,
                                                        max_lr=cfg_sch.lr_max, mode=cfg_sch.mode, gamma=cfg_sch.gamma,
                                                        cycle_momentum=cfg_sch.cycle_momentum)
            elif name_sch == 'multisteplr':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg_sch.milestones,
                                                           gamma=cfg_sch.gamma)
            elif name_sch == 'exponentiallr':
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg_sch.gamma)
            elif name_sch == 'lambdalr':
                lambdas_lr = []
                for where in cfg_sch.where: # e.g., model.lambda_lr_0
                    attrs = where.split('.')
                    lambda_lr = self
                    for attr in attrs:
                        lambda_lr = getattr(lambda_lr, attr)
                    lambdas_lr.append(lambda_lr)
                scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambdas_lr)
            else:
                raise NotImplementedError(f'Unknown scheduler: {name_sch}')
        else:
            scheduler = None

        return optimizer, scheduler
    
    def _train_epochs(self):

        for epoch in range(self.epoch_start, self.epoch_end):
            self.epoch_total += 1
            # validation
            if epoch == self.cfg.exp.train.epoch_start:
                if self.cfg.exp.val.skip_initial_val:
                    skip_val = True
                else:
                    skip_val = False
            elif self.epoch_total % self.cfg.exp.val.n_epochs_once != 0:
                skip_val = True
            elif self.epoch_total < self.cfg.exp.val.no_val_before_epoch:
                skip_val = True
            else:
                skip_val = False
            if not skip_val:
                if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                    self.val(self.epoch_total, mode='val')
                    if self.is_best:
                        self.val(self.epoch_total, mode='test')
            if self.cfg.var.is_parallel:
                dist.barrier()

            self.model.train()
            self.model.before_epoch(mode='train', i_repeat=self.epoch_total)

            if self.cfg.var.is_parallel:
                # see WARNING in https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
                self.sampler_train.set_epoch(self.epoch_total)

            if self.cfg.model.task_sequential:
                print(f'----------- task-{self.task_idx} training epoch begins -----------')
            else:
                print(f'----------- training epoch begins -----------')
            if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                if self.cfg.model.task_sequential:
                    obj_to_enumerate = track(self.train_loaders[self.task_idx], transient=True, description='training')
                else:
                    obj_to_enumerate = track(self.train_loader, transient=True, description='training')
            else:
                if self.cfg.model.task_sequential:
                    obj_to_enumerate = self.train_loaders[self.task_idx]
                else:
                    obj_to_enumerate = self.train_loader
                
            for _, data in enumerate(obj_to_enumerate):
                self.iter_total += 1

                if not self.cfg.exp.customize_dataloader:
                    if self.cfg.model.task_sequential:
                        if hasattr(self.train_sets[self.task_idx], 'to_device'):
                            data = self.train_sets[self.task_idx].to_device(data, device=self.device)
                        else:
                            input, ground_truth = data
                            data = input.to(self.device), ground_truth.to(self.device)
                    else:
                        if hasattr(self.train_set, 'to_device'):
                            data = self.train_set.to_device(data, device=self.device)
                        else:
                            input, ground_truth = data
                            data = input.to(self.device), ground_truth.to(self.device)
                
                if self.cfg.model.task_sequential:
                    output = self.model.observe(data)
                    metrics = self.model.metrics
                else:
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    metrics = self.model.get_metrics(data, output, mode='train')
                    if self.cfg.exp.train.use_gradscaler:
                        self.gradscaler.scale(metrics['loss_final']).backward()
                        self.gradscaler.unscale_(self.optimizer)
                        self.gradscaler.step(self.optimizer)
                        self.gradscaler.update()
                    else:
                        metrics['loss_final'].backward()
                        self.optimizer.step()

                for name, value in metrics.items():
                    if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                        self.writer.add_scalar(f'train/{name}', value, self.iter_total)
                
                # validation
                if self.iter_total % self.cfg.exp.val.n_iters_once == 0:
                    if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                        self.val(self.iter_total, mode='val', val_iter=True)
                        if self.is_best:
                            self.val(self.iter_total, mode='test', val_iter=True)
                if self.cfg.var.is_parallel:
                    dist.barrier()

                self.model.train()

            self.model.after_epoch(mode='train')

            if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                self.model.vis(self.writer, self.epoch_total, data, output, mode='train', in_epoch=False)

            if hasattr(self, 'scheduler'):
                if self.scheduler:
                    self.scheduler.step()

            result_log = [f'epoch: {self.epoch_total}']
            if self.cfg.model.task_sequential:
                result_log += [f'task: {self.cfg.dataset.train_tasks[self.task_idx]}']
            for name, value in self.model.metrics_epoch_train.items():
                result_log.append(f'{name}: {value:.4f}')
            info_logged = ', '.join(result_log)
            self.logger_extra.warn(f'[train] {info_logged}')
            self.logger_train.info(info_logged)
            
        return self.iter_total, self.epoch_total

    def train(self):
        self.logger_extra.warn(f'------ Training ------')
        self.model = self.model.to(self.device)
        if self.cfg.var.is_parallel:
            id_device = self.cfg.exp.idx_device[dist.get_rank()]
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = MyDistributedDataParallel(self.model, device_ids=[id_device])
        if self.cfg.exp.train.use_gradscaler:
            self.cfg.var.gradscaler = torch.cuda.amp.GradScaler()

        if self.cfg.model.task_sequential:
            pass
        else:
            self.optimizer, self.scheduler = self._get_optimizer(getattr(self.model, 'get_params', self.model.parameters)())

        if self.cfg.exp.train.path_model_trained is not None:
            if self.cfg.var.is_parallel:
                dist.barrier()
                id_device = self.cfg.exp.idx_device[dist.get_rank()]
                map_location = {'cuda:0': f'cuda:{id_device}'}
            else:
                map_location = self.device
            self.model.load_state_dict(torch.load(self.cfg.exp.train.path_model_trained, map_location=map_location),
                                       strict=True)
       
        if hasattr(self.model, 'before_train'):
            self.model.before_train()

        self.epoch_total = 0
        self.iter_total = 0
        if self.cfg.model.task_sequential:
            if isinstance(self.cfg.exp.train.n_epochs, (float, int)):
                self.n_epochs = [self.cfg.exp.train.n_epochs] * len(self.cfg.dataset.train_tasks)
            elif isinstance(list(self.cfg.exp.train.n_epochs), list):
                assert len(self.cfg.exp.train.n_epochs) == len(self.cfg.dataset.train_tasks)
                self.n_epochs = list(self.cfg.exp.train.n_epochs)
            else:
                raise NotImplementedError
            self.n_epochs.insert(0, 0)
            self.cum_epochs = np.cumsum(self.n_epochs) + self.cfg.exp.train.epoch_start
            for task_idx in range(len(self.cfg.dataset.train_tasks)):
                self.epoch_start = self.cum_epochs[task_idx]
                self.epoch_end = self.cum_epochs[task_idx + 1]
                self.task_idx = task_idx
                self.score_best = -math.inf
                self.score_best_test = -math.inf
                self.is_best = True
                
                if hasattr(self.model, 'begin_task'):
                    self.model.begin_task(self.train_loaders[task_idx])
                
                _, epoch = self._train_epochs()
                
                if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                    self.val(epoch, mode='val')
                    self.val(epoch, mode='test')
                
                if hasattr(self.model, 'end_task'):
                    self.model.end_task(self.train_loaders[task_idx])     
                
        else:   
            self.epoch_start = self.cfg.exp.train.epoch_start
            self.epoch_end = self.epoch_start + self.cfg.exp.train.n_epochs
            self.score_best = -math.inf
            self.score_best_test = -math.inf
            self.is_best = True
            self._train_epochs()
            
        self.logger_train.warn(f'------ Training finished ------')

        if hasattr(self.model, 'after_train'):
            self.model.after_train()

        if self.cfg.var.is_parallel:
            dist.destroy_process_group()

    def val(self, epoch, mode='val', val_iter=False):
        assert mode in ['val', 'test']
        self.is_best = False
        self.is_best_test = False
        if self.cfg.model.task_sequential:
            data_loader = getattr(self, f'{mode}_loaders')[self.task_idx]
            dataset = getattr(self, f'{mode}_sets')[self.task_idx]
        else:
            data_loader = getattr(self, f'{mode}_loader')
            dataset = getattr(self, f'{mode}_set')
            
        mark = 'iter' if val_iter else 'epoch'
        task_suffix = f'_task{self.task_idx}' if self.cfg.model.task_sequential else ''

        with torch.no_grad():
            self.model.eval()

            for i_repeat in range(self.cfg.exp[mode].n_repeat):
                self.model.before_epoch(mode, i_repeat)
                if self.cfg.model.task_sequential:
                    print(f'----------- task-{self.task_idx} {mode} epoch begins -----------')
                else:
                    print(f'----------- {mode} epoch begins -----------')
                if val_iter:
                    obj_to_enumerate = data_loader
                else:
                    obj_to_enumerate = track(data_loader, transient=True, description=mode)
                for _, data in enumerate(obj_to_enumerate):
                    if not self.cfg.exp.customize_dataloader:
                        if hasattr(dataset, 'to_device'):
                            data = dataset.to_device(data, device=self.device)
                        else:
                            input, ground_truth = data
                            data = input.to(self.device), ground_truth.to(self.device)
                    if self.cfg.var.is_parallel and dist.get_rank() == 0:
                        # if we use self.model(data), then after validation, training will get stuck
                        # see https://github.com/pytorch/pytorch/issues/54059#issuecomment-801754630
                        output = self.model.module(data)
                    else:
                        output = self.model(data)
                    self.model.get_metrics(data, output, mode=mode)
                    if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                        self.model.vis(self.writer, epoch, data, output, mode=mode, in_epoch=True)
                self.model.after_epoch(mode)

                if mode == 'val':
                    self.is_best = self.model.metrics_epoch_val['metric_final'] > self.score_best
                    if self.is_best:
                        self.score_best = self.model.metrics_epoch_val['metric_final']
                elif mode == 'test' and self.cfg.exp.mode == 'train':
                    self.is_best_test = self.model.metrics_epoch_val['metric_final'] > self.score_best_test
                    if self.is_best_test:
                        self.score_best_test = self.model.metrics_epoch_val['metric_final']

                result_log = [f'{mark}: {epoch}']
                if self.cfg.model.task_sequential:
                    if mode in ['train', 'val']:
                        result_log += [f'task: {self.cfg.dataset.train_tasks[self.task_idx]}']
                    else:
                        result_log += [f'task: {self.cfg.dataset.test_tasks[self.task_idx]}']

                for (name, value) in self.model.metrics_epoch_val.items():
                    result_log.append(f'{name}: {value:.4f}')
                info_logged = ', '.join(result_log)
                self.logger_extra.warn(f'[{mode}] {info_logged}')
                self.logger_val.info(info_logged)

            if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                self.model.vis(self.writer, epoch, data, output, mode=mode, in_epoch=False)

            for (name, value) in self.model.metrics_epoch_val.items():
                if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                    self.writer.add_scalar(f'{mode}/{name}', value, epoch)

            # save best model
            if mode == 'val' and self.is_best:
                if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                    self.logger_checkpoints.warn(f'Saving best model on val set: {mark} {epoch}')
                    torch.save(self.model.state_dict(), os.path.join(self.path_checkpoints, f'model_best_val{task_suffix}.pth'))
            if mode == 'test' and self.is_best_test and self.cfg.exp.train.save_best_model_on_test_set:
                if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                    self.logger_checkpoints.warn(f'Saving best model on test set: {mark} {epoch}')
                    torch.save(self.model.state_dict(), os.path.join(self.path_checkpoints, f'model_best_test{task_suffix}.pth'))

            if mode == 'val':
                if getattr(self.cfg.exp.val, 'save_every_model', False):
                    self.logger_checkpoints.warn(f'Saving current model: {mark} {epoch}')
                    torch.save(self.model.state_dict(), os.path.join(self.path_checkpoints, f'model_{mark}{epoch}{task_suffix}.pth'))
                elif self.is_best and getattr(self.cfg.exp.val, 'save_every_better_model', False):
                    self.logger_checkpoints.warn(f'Saving current model: {mark} {epoch}')
                    torch.save(self.model.state_dict(), os.path.join(self.path_checkpoints, f'model_{mark}{epoch}{task_suffix}.pth'))
                if self.cfg.exp.val.save_latest_model:
                    self.logger_checkpoints.warn(f'Saving latest model: {mark} {epoch}')
                    torch.save(self.model.state_dict(), os.path.join(self.path_checkpoints, f'model_latest{task_suffix}.pth'))

    def test(self):
        self.model = self.model.to(self.device)

        if self.cfg.exp.test.path_model_trained is None:
            print('warning: no model is loaded')
        else:
            dict_state = torch.load(self.cfg.exp.test.path_model_trained, map_location=self.device)
            for key in list(dict_state.keys()):
                if key.startswith('module.'):
                    dict_state[key[7:]] = dict_state.pop(key)
            self.model.load_state_dict(dict_state, strict=False)

        for key in list(dict_state.keys()):
            if key.startswith('module.'):
                dict_state[key[7:]] = dict_state.pop(key)
        print(f'loading pretrained model for test from path {self.cfg.exp.test.path_model_trained}')
        self.model.load_state_dict(dict_state, strict=True)

        if self.cfg.model.task_sequential:
            for task_idx in range(len(self.cfg.dataset.test_tasks)):
                self.task_idx = task_idx
                self.val(epoch=0, mode='test')
        else:
            self.val(epoch=0, mode='test')

        if self.cfg.var.is_parallel:
            dist.destroy_process_group()
