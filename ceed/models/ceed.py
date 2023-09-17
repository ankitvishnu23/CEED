import argparse
import numpy as np
import os 
import sys
import subprocess
import random

import torch
import torch.backends.cudnn as cudnn

from types import SimpleNamespace
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, WFDataset_lab
from data_aug.wf_data_augs import Crop
from ceed.models.model_simclr import ModelSimCLR, Projector, Projector2
from utils.utils import get_torch_reps
from utils.load_models import load_ckpt_to_model
from ceed.simclr import SimCLR
from ceed.models.model_GPT import GPTConfig, Single_GPT, Multi_GPT
from analysis.encoder_utils import load_GPT_backbone
from torch.nn.parallel import DistributedDataParallel as DDP

class CEED(object):
    """
        CEED object that can be trained, loaded from checkpoint, and can be used to get low-d representations of ephys data.
    """
    def __init__(
        self,
        model_arch: str = "fc_encoder",
        out_dim: int = 5,
        proj_dim: int = 5,
        num_extra_chans: int = 0,
        gpu: int = 0
    ):
        """
        Parameters
        ----------
        model_arch : str
            The type of model being trained or loaded (fc_encoder, attention_enc)
        out_dim : int
            The size of the representation layer before the projection head
        proj_dim : int
            The size of the representation layer after the projection head
        num_extra_chans : int
            The number of channels to use on each side of the max amplitude channel for training and getting representations
        gpu : int
            The index of the cuda device being used by the CEED model
        """

        self.multi_chan = True if num_extra_chans > 1 else False
        self.ddp = True if model_arch == "gpt" else False
        self.out_dim = out_dim
        self.num_extra_chans = num_extra_chans
        self.arch = model_arch

        if self.arch == "gpt":
            model_args = dict(n_layer=20, n_head=4, n_embd=64, block_size=121*out_dim,
                    bias=True, vocab_size=50304, dropout=0.0, out_dim=out_dim, is_causal=True, 
                    proj_dim=proj_dim, pos='seq_11times', multi_chan=self.multi_chan) 
            gptconf = GPTConfig(**model_args)
            if self.multi_chan:
                self.model = Multi_GPT(gptconf).cuda(gpu)
            else:
                self.model = Single_GPT(gptconf).cuda(gpu)
        else:
            self.model = ModelSimCLR(base_model=self.arch, out_dim=out_dim, proj_dim=proj_dim, 
                fc_depth=2, expand_dim=False, multichan=self.multi_chan, input_size=(2*num_extra_chans+1)*121).cuda(gpu)


    def train(
        self,
        data_dir,
        exp_name,
        log_dir,
        ckpt_dir,
        gpu: int = 0,
        epochs: int = 400,
        lr: float = 0.001,
        batch_size: int = 256,
        optimizer: str = 'adam',
        cell_type: bool = False,
        save_metrics: bool = False,
        use_chan_pos: bool = False,
    ):
        """Trains a CEED model

        Parameters
        ----------
        data_dir : str
            The absolute path location of the CEED neural ephys dataset.
        exp_name : str
            The name of the experiment - folder with this name will be created in log_dir and ckpt_dir. 
        log_dir : str
            The absolute path location to which logs will be stored. 
        ckpt_dir : str
            The absolute path location to which ckpts will be saved or from which ckpts will be restored. 
        gpu : int
            The index of the cuda device being used by the CEED model.
        epochs : int
            The number of epochs to train the CEED model.
        lr : float
            The learning rate used for the optimizer.
        batch_size : int
            The number of samples per batch to contrastively train CEED (higher bs can improve performance but requires more GPU memory).
        optimizer : str
            'adam' or 'sgd' optimizer to use for training.
        cell_type : bool
            Whether to normalize data for use in training a CEED cell type classification model.
        save_metrics : bool
            Whether to run CEED on a test/val set after every epoch and chart performance. 
        use_chan_pos : bool
            Whether to use channel location data (x, y on probe) to train CEED. 
        """

        checkpoint_dir = os.path.join(ckpt_dir, exp_name, 'test')
        log_dir = os.path.join(log_dir, exp_name, 'test')
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        torch.backends.cudnn.benchmark = True

        dataset = ContrastiveLearningDataset(data_dir, self.out_dim, multi_chan=self.multi_chan)

        train_dataset = dataset.get_dataset('wfs', 2, 1.0, self.num_extra_chans, normalize=cell_type, detected_spikes=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True, drop_last=True)
        
        if save_metrics:
            if self.multi_chan:
                memory_dataset = WFDataset_lab(data_dir, split='train', multi_chan=self.multi_chan, transform=Crop(prob=0.0, num_extra_chans=self.num_extra_chans, ignore_chan_num=True), use_chan_pos=use_chan_pos)
                memory_loader = torch.utils.data.DataLoader(
                    memory_dataset, batch_size=128, shuffle=False,
                    num_workers=8, pin_memory=True, drop_last=False)
                test_dataset = WFDataset_lab(data_dir, split='test', multi_chan=self.multi_chan, transform=Crop(prob=0.0, num_extra_chans=self.num_extra_chans, ignore_chan_num=True), use_chan_pos=use_chan_pos)
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=8, pin_memory=True, drop_last=False)
            else:
                memory_dataset = WFDataset_lab(data_dir, split='train', multi_chan=False)
                memory_loader = torch.utils.data.DataLoader(
                    memory_dataset, batch_size=128, shuffle=False,
                    num_workers=8, pin_memory=True, drop_last=False)
                test_dataset = WFDataset_lab(data_dir, split='test', multi_chan=False)
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=8, pin_memory=True, drop_last=False)
        else:
            memory_loader = None
            test_loader = None
        

        print("number of transfomer params: ", sum(p.numel() for n,p in self.model.named_parameters() if 'transformer' in n))
        print("number of fcpart params: ", sum(p.numel() for n,p in self.model.named_parameters() if ('lm_head' in n and 'proj' not in n)))
        print("number of Proj params: ", sum(p.numel() for n,p in self.model.named_parameters() if ('proj' in n)))
        print("number of online classifier params: ", sum(p.numel() for n,p in self.model.named_parameters() if 'online_head' in n))

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=1e-4)
            scheduler = None
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(train_loader)), eta_min=0,
                                                                last_epoch=-1)

        print("model and optimizer initialized!")
        # automatically resume from checkpoint if it exists
        if os.path.exists(os.path.join(checkpoint_dir, "checkpoint.pth")):
            print("loading from previous checkpoint: ", checkpoint_dir)
            ckpt = torch.load(os.path.join(checkpoint_dir, "checkpoint.pth"),
                            map_location='cpu')
            start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])

        else:
            start_epoch = 0

        args = SimpleNamespace(data=data_dir, ddp=False, rank=0, log_dir=log_dir, 
                               multi_chan=self.multi_chan, n_views=2, 
                               fp16=True, epochs=epochs, add_train=True, 
                               use_chan_pos=use_chan_pos, use_gpt=self.ddp,
                               online_head=False, eval_knn_every_n_epochs=1,
                               no_knn=(not save_metrics), checkpoint_dir=checkpoint_dir, 
                               num_extra_chans=self.num_extra_chans, 
                               disable_cuda=False, temperature=0.07, arch=self.arch,
                               noise_scale=1.0, cell_type=cell_type, gpu=gpu)
        
        print("starting training...")
    
        simclr = SimCLR(model=self.model, proj=None, optimizer=optimizer, scheduler=scheduler, gpu=gpu, 
                        sampler=None, args=args, start_epoch=start_epoch)
        simclr.train(train_loader, memory_loader, test_loader)


    def load(self, ckpt_dir):
        """ Loads CEED from a checkpoint
        
        Parameters
        ----------
        ckpt_dir : str
            The absolute path location from which ckpt will be restored. 
        """
        checkpoint_dir = os.path.join(ckpt_dir, 'test')
        print("loading from previous checkpoint: ", checkpoint_dir)
        ckpt = os.path.join(checkpoint_dir, "checkpoint.pth")
        if self.ddp:
            load_ckpt_to_model(self.model, ckpt, self.multi_chan)
        else:
            self.model.backbone.load(ckpt)
    

    def transform(self, data_dir, use_chan_pos=False, file_split='test'):
        """ Loads CEED from a checkpoint
        
        Parameters
        ----------
        data_dir : str
            The absolute path location from which neural ephys data will be loaded into CEED to obtain representations.
        use_chan_pos: bool
            Whether channel locations (x, y on the probe) will be used to obtain representations (only if CEED model was trained using channel locations).
        file_split: str
            Which data split to transform - 'test', 'val', or 'train'. Will look for corresponding spikes file. 
        """
        if self.multi_chan:
            dataset = WFDataset_lab(data_dir, split=file_split, multi_chan=self.multi_chan, transform=Crop(prob=0.0, num_extra_chans=self.num_extra_chans, ignore_chan_num=True), use_chan_pos=use_chan_pos)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=128, shuffle=False,
                num_workers=8, pin_memory=True, drop_last=False)
        else:
            dataset = WFDataset_lab(data_dir, split=file_split, multi_chan=False)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=128, shuffle=False,
                num_workers=8, pin_memory=True, drop_last=False)
            
        args = SimpleNamespace(ddp=False, rank=0,
                multi_chan=self.multi_chan, use_chan_pos=use_chan_pos, 
                use_gpt=self.ddp, num_extra_chans=self.num_extra_chans)
        
        reps_test, labels_test = get_torch_reps(self.model, loader, 0, args)

        return reps_test, labels_test
