import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from utils.utils import (
    save_checkpoint, knn_monitor, gmm_monitor, save_reps
)      
# from utils.ddp_utils import gather_from_all
import tensorboard_logger as tb_logger
torch.manual_seed(0)
import time
from data_aug.wf_data_augs import TorchSmartNoise

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.gpu = kwargs['gpu']
        self.sampler = kwargs['sampler']
        
        self.model =  kwargs['model']
        self.proj = kwargs['proj'].cuda(kwargs['gpu']) if kwargs['proj'] is not None else None 
        if self.proj and self.args.ddp:
            raise "proj needs to be wrapped in ddp"
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        if self.args.rank == 0 or not self.args.ddp:
            self.logger = tb_logger.Logger(logdir=self.args.log_dir, flush_secs=2)
        self.multichan = self.args.multi_chan
        if self.args.rank == 0 or not self.args.ddp:
            logging.basicConfig(filename=os.path.join(self.args.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.gpu)
        self.start_epoch = kwargs['start_epoch']
        self.noise_transform = TorchSmartNoise(self.args.data,
                                               noise_scale=self.args.noise_scale,
                                               normalize=self.args.cell_type, 
                                               gpu=self.gpu,
                                               p=self.args.aug_p_dict[-1])

    def info_nce_loss(self, features):
        # if self.args.ddp:
        #     features = gather_from_all(features)
        features = torch.squeeze(features)
        features = F.normalize(features, dim=1)
        batch_dim = int(features.shape[0] // 2)

        labels = torch.cat([torch.arange(batch_dim) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(non_blocking=True)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, memory_loader=None, test_loader=None):

        scaler = GradScaler(enabled=self.args.fp16)

        n_iter = 0
        if self.args.rank == 0 or not self.args.ddp:
            logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
            logging.info(f"Training with gpu: {not self.args.disable_cuda}.")
            print(f"Start SimCLR training for {self.args.epochs} epochs, starting at {self.start_epoch}.")

        for epoch_counter in range(self.start_epoch, self.args.epochs):
            # if self.args.add_train:
            self.model.train()
            start_time = time.time()
            if self.args.ddp:
                self.sampler.set_epoch(epoch_counter)
            print('Epoch {}'.format(epoch_counter))
            for i, (wf, chan_nums, lab) in enumerate(train_loader):
                chan_pos = None
                if self.args.use_chan_pos:
                    wf, chan_pos = wf
                    chan_pos = torch.cat(chan_pos, dim=0).float()
                wf = torch.cat(wf, dim=0).float()
                chan_nums = np.concatenate(chan_nums, axis=0)
                lab = torch.cat(lab, dim=0).long().cuda(self.gpu,non_blocking=True)
                
                wf = wf.cuda(self.gpu)
                wf = self.noise_transform([wf, chan_nums]) #smart_noise on GPU

                if self.args.use_gpt:
                    if not self.args.multi_chan:
                        wf = torch.squeeze(wf, dim=1)
                        wf = torch.unsqueeze(wf, dim=-1)
                    else:
                        wf = wf.view(-1, (self.args.num_extra_chans*2+1)*121)
                        wf = torch.unsqueeze(wf, dim=-1)
                
                with autocast(enabled=self.args.fp16):
                    if self.args.online_head:
                        features, cls_loss, online_acc = self.model(wf, lab, chan_pos=chan_pos)
                    else:
                        if self.args.use_gpt:
                            features = self.model(wf, chan_pos=chan_pos)
                        else:
                            features = self.model(wf)
                        cls_loss = 0.
                        online_acc = -1
                    if self.proj is not None:
                        features = self.proj(features)
                    
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels) + cls_loss
                    
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                
                n_iter += 1

                # warmup for the first 10 epochs
                if epoch_counter >= 10 and self.scheduler != None:
                    self.scheduler.step()
            
            if epoch_counter % self.args.eval_knn_every_n_epochs == 0 and epoch_counter != 0 and not self.args.no_knn:
                if self.args.rank == 0 or not self.args.ddp:
                    knn_score = knn_monitor(net=self.model, memory_data_loader=memory_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=self.args)
                    gmm_score = gmm_monitor(net=self.model, memory_data_loader=memory_loader, test_data_loader=test_loader, device='cuda', epoch_num=epoch_counter, hide_progress=True, args=self.args)
                    print(f"Epoch {epoch_counter}, my knn_acc:{knn_score}")
                    print(f"Epoch {epoch_counter}, my gmm_acc:{gmm_score}")
                    self.logger.log_value('knn_acc', knn_score, epoch_counter)
                    self.logger.log_value('gmm_acc', gmm_score, epoch_counter)
                    
            if self.args.rank == 0 or not self.args.ddp:
                logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}")
                self.logger.log_value('loss', loss, epoch_counter)
                
                curr_lr = self.optimizer.param_groups[0]['lr'] if self.scheduler == None else self.scheduler.get_lr()[0]
                self.logger.log_value('learning_rate', curr_lr, epoch_counter)
                if online_acc != -1:
                    self.logger.log_value('online_acc', online_acc, epoch_counter)
                    print("loss: ", loss.item(), "online acc: ", online_acc)
            
            print(f"time for epoch {epoch_counter}: {time.time()-start_time}")
            if self.args.rank == 0 or not self.args.ddp:
                # save model checkpoints
                save_dict = {
                    'epoch': epoch_counter,
                    'arch': self.args.arch,
                    'optimizer': self.optimizer.state_dict(),
                    'state_dict': self.model.state_dict()
                    }
                    
                save_checkpoint(save_dict, is_best=False, filename=os.path.join(self.args.checkpoint_dir, 'checkpoint.pth'))
                print(f"Model checkpoint and metadata has been saved at {self.args.checkpoint_dir}.")
                logging.info(f"Model checkpoint and metadata has been saved at {self.args.checkpoint_dir}.")

        if self.args.rank == 0 or not self.args.ddp:
            logging.info("Training has finished.")
            # save model checkpoints
            # checkpoint_name = self.args.exp + '_checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
            save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(self.args.checkpoint_dir, 'final.pth'))
            logging.info(f"Model checkpoint and metadata has been saved at {self.args.checkpoint_dir}.")
        
            # save out the representations of the test and memory loaders using the final checkpoint
            save_reps(self.model, test_loader, os.path.join(self.args.checkpoint_dir, 'final.pth'), split='test', multi_chan=False, rep_after_proj=True, use_chan_pos=False, suffix='')
            save_reps(self.model, memory_loader, os.path.join(self.args.checkpoint_dir, 'final.pth'), split='test', multi_chan=False, rep_after_proj=True, use_chan_pos=False, suffix='')
            
            