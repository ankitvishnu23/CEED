import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn

from types import SimpleNamespace
from torchvision import models
from data_aug.contrastive_learning_dataset import (
    ContrastiveLearningDataset,
    WFDataset_lab,
)
from data_aug.wf_data_augs import Crop
from ceed.models.model_simclr import ModelSimCLR
from utils.utils import get_torch_reps, get_torch_reps_nolabels, apply_transform
from utils.load_models import load_ckpt_to_model
from ceed.simclr import SimCLR
from ceed.models.model_SCAM import SCAMConfig, Multi_SCAM
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
        gpu: int = 0,
        old_ckpt: bool = False,
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
        old_ckpt : bool
            Checkpoints older than 11/1/2023 are loaded differently (to be deprecated)
        """

        self.multi_chan = True if num_extra_chans > 1 else False
        self.ddp = True if model_arch == "scam" else False
        self.out_dim = out_dim
        self.num_extra_chans = num_extra_chans
        self.arch = model_arch
        self.num_classes = 400
        if gpu is None:
            self.device = "cpu"
        else:
            self.device = gpu

        if self.arch == "scam":
            model_args = dict(
                n_layer=20,
                n_head=4,
                n_embd=64,
                block_size=121 * (2 * self.num_extra_chans + 1),
                bias=True,
                vocab_size=50304,
                dropout=0.0,
                out_dim=out_dim,
                proj_dim=proj_dim,
                multi_chan=self.multi_chan,
                num_classes=self.num_classes,
            )
            scamconf = SCAMConfig(**model_args)
            self.model = Multi_SCAM(scamconf)
            
        else:
            self.model = ModelSimCLR(
                base_model=self.arch,
                out_dim=out_dim,
                proj_dim=proj_dim,
                fc_depth=2,
                expand_dim=False,
                multichan=self.multi_chan,
                input_size=(2 * num_extra_chans + 1) * 121,
                old_ckpt=old_ckpt,
            )
        if gpu is not None:
            self.model = self.model.to(gpu)

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
        optimizer: str = "adam",
        aug_p_dict: dict = {
            "collide": 0.4,
            "crop_shift": 0.4,
            "amp_jitter": 0.5,
            "temporal_jitter": 0.7,
            "smart_noise": (0.6, 1.0),
        },
        cell_type: bool = False,
        save_metrics: bool = False,
        n_units: int = 10,
        units_list: list = None,
    ):
        """Trains a CEED model with the MLP backbone

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
        aug_p_dict : list
            probability of using augmentation in stochastic aug pipeline.
            order of aug probabilities is [collision, crop, amplitude jitter, temporal jitter, smart noise]
        cell_type : bool
            Whether to normalize data for use in training a CEED cell type classification model.
        save_metrics : bool
            Whether to run CEED on a test/val set after every epoch and chart performance.
        n_units : int
            The number of units to subselect from the dataset and compute metrics on.
        units_list : list
            (Optional) List of units to select from the dataset and on which to compute metrics.
        """

        if units_list is not None:
            n_units = len(units_list)

        checkpoint_dir = os.path.join(ckpt_dir, exp_name)
        log_dir = os.path.join(log_dir, exp_name)

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        torch.backends.cudnn.benchmark = True

        dataset = ContrastiveLearningDataset(
            data_dir, self.out_dim, multi_chan=self.multi_chan
        )

        train_dataset = dataset.get_dataset(
            name="wfs",
            n_views=2,
            num_extra_chans=self.num_extra_chans,
            detected_spikes=False,
            aug_p_dict=aug_p_dict,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        if self.multi_chan:
            memory_dataset = WFDataset_lab(
                data_dir,
                split="train",
                multi_chan=self.multi_chan,
                transform=Crop(
                    prob=0.0, num_extra_chans=self.num_extra_chans, ignore_chan_num=True
                ),
                n_units=n_units,
                units_list=units_list,
            )
            memory_loader = torch.utils.data.DataLoader(
                memory_dataset,
                batch_size=128,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
            )
            test_dataset = WFDataset_lab(
                data_dir,
                split="test",
                multi_chan=self.multi_chan,
                transform=Crop(
                    prob=0.0, num_extra_chans=self.num_extra_chans, ignore_chan_num=True
                ),
                n_units=n_units,
                units_list=units_list,
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
            )
        else:
            memory_dataset = WFDataset_lab(
                data_dir,
                split="train",
                multi_chan=False,
                n_units=n_units,
                units_list=units_list,
            )
            memory_loader = torch.utils.data.DataLoader(
                memory_dataset,
                batch_size=128,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
            )

            test_dataset = WFDataset_lab(
                data_dir,
                split="test",
                multi_chan=False,
                n_units=n_units,
                units_list=units_list,
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
            )
        print(
            "number of transfomer params: ",
            sum(
                p.numel()
                for n, p in self.model.named_parameters()
                if "transformer" in n
            ),
        )
        print(
            "number of fcpart params: ",
            sum(
                p.numel()
                for n, p in self.model.named_parameters()
                if ("lm_head" in n and "proj" not in n)
            ),
        )
        print(
            "number of Proj params: ",
            sum(p.numel() for n, p in self.model.named_parameters() if ("proj" in n)),
        )
        print(
            "number of online classifier params: ",
            sum(
                p.numel()
                for n, p in self.model.named_parameters()
                if "online_head" in n
            ),
        )

        if optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=1e-4)
            scheduler = None
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(epochs * len(train_loader)), eta_min=0, last_epoch=-1
            )

        print("model and optimizer initialized!")
        # automatically resume from checkpoint if it exists
        if os.path.exists(os.path.join(checkpoint_dir, "checkpoint.pth")):
            print("loading from previous checkpoint: ", checkpoint_dir)
            ckpt = torch.load(
                os.path.join(checkpoint_dir, "checkpoint.pth"), map_location="cpu"
            )
            start_epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])

        else:
            start_epoch = 0

        args = SimpleNamespace(
            data=data_dir,
            ddp=False,
            rank=0,
            log_dir=log_dir,
            multi_chan=self.multi_chan,
            n_views=2,
            fp16=True,
            epochs=epochs,
            add_train=True,
            use_gpt=self.ddp,
            online_head=False,
            eval_knn_every_n_epochs=1,
            no_knn=(not save_metrics),
            checkpoint_dir=checkpoint_dir,
            num_extra_chans=self.num_extra_chans,
            disable_cuda=False,
            temperature=0.07,
            arch=self.arch,
            noise_scale=aug_p_dict["smart_noise"][1],
            cell_type=cell_type,
            gpu=gpu,
            aug_p_dict=aug_p_dict,
        )
        print(aug_p_dict)
        print(aug_p_dict["smart_noise"][1])

        print("starting training...")
        simclr = SimCLR(
            model=self.model,
            proj=None,
            optimizer=optimizer,
            scheduler=scheduler,
            gpu=gpu,
            sampler=None,
            args=args,
            start_epoch=start_epoch,
        )
        simclr.train(train_loader, memory_loader, test_loader)

    def load(self, ckpt_dir):
        """Loads CEED from a checkpoint

        Parameters
        ----------
        ckpt_dir : str
            The absolute path location from which ckpt will be restored.
        """
        #checkpoint_dir = os.path.join(ckpt_dir, "test")
        print("loading from previous checkpoint: ", ckpt_dir)
        ckpt = os.path.join(ckpt_dir, "checkpoint.pth")
        if self.ddp:
            load_ckpt_to_model(self.model, ckpt, self.multi_chan)
        else:
            self.model.backbone.load(ckpt)

    def load_and_transform(self, data_dir, units_list=None, file_split="test"):
        """Load a spike dataset from a folder and transform the data

        Parameters
        ----------
        data_dir : str
            The absolute path location from which neural ephys data will be loaded into CEED to obtain representations.
        units: list
            List of unit ids to load and transform. If None, then all units will be loaded and transformed.
        file_split: str
            Which data split to transform - 'test', 'val', or 'train'. Will look for corresponding spikes file.
        """
        if units_list is None:
            n_units = -1 #load and transform all units
        else:
            n_units = len(units_list)
        if self.multi_chan:
            dataset = WFDataset_lab(
                data_dir,
                split=file_split,
                multi_chan=self.multi_chan,
                transform=Crop(
                    prob=0.0, num_extra_chans=self.num_extra_chans, ignore_chan_num=True
                ),
                n_units=n_units,
                units_list=units_list,
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
            )
        else:
            dataset = WFDataset_lab(data_dir, 
                                    split=file_split, 
                                    multi_chan=False, 
                                    n_units=n_units, 
                                    units_list=units_list
                                   )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
            )

        args = SimpleNamespace(
            ddp=False,
            rank=0,
            multi_chan=self.multi_chan,
            use_gpt=self.ddp,
            num_extra_chans=self.num_extra_chans,
            arch=self.arch,
        )

        reps_test, labels_test = get_torch_reps(self.model, loader, self.device, args)

        return reps_test, labels_test

    def transform(self, data):
        """Transform data using CEED model

        Parameters
        ----------
        data: numpy.ndarray
            A collection of spike data formatted as such (N, spike_length_samples)
        """
        if self.multi_chan:
            crop_tform = Crop(
                prob=0.0, num_extra_chans=self.num_extra_chans, ignore_chan_num=True
            )
            data = apply_transform(transform=crop_tform, data=data)
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=128,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

        args = SimpleNamespace(
            ddp=False,
            rank=0,
            multi_chan=self.multi_chan,
            use_gpt=self.ddp,
            num_extra_chans=self.num_extra_chans,
            arch=self.arch,
        )

        reps_test = get_torch_reps_nolabels(self.model, loader, self.device, args)

        return reps_test
