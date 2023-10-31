from distutils.command.build import build
from pathlib import Path
import argparse
import os
import sys
import random

import time
import json
import math
import numpy as np

from torch import nn, optim
import torch
import torch.distributed as dist

import tensorboard_logger as tb_logger

sys.path.append('../')      
sys.path.append('../..')    
sys.path.append('.')  

from utils.ddp_utils import gather_from_all

from data_aug.contrastive_learning_dataset import (
    ContrastiveLearningDataset,
    WFDataset_lab,
)
from ceed.models.model_SCAM import SCAMConfig, Multi_SCAM
from ceed.models.model_simclr import ModelSimCLR, Projector
from data_aug.wf_data_augs import TorchSmartNoise

from utils.utils import knn_monitor, gmm_monitor, save_reps
from data_aug.wf_data_augs import Crop

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main_worker(gpu, args):
    fix_seed(args.seed)

    if args.ddp:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

        logger = tb_logger.Logger(logdir=args.log_dir, flush_secs=2)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = SimCLR(args).cuda(gpu)

    if args.ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=True
        )

    optimizer = torch.optim.Adam(
            model.parameters(), args.learning_rate, weight_decay=args.weight_decay
        )

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / "checkpoint.pth").is_file():
        ckpt = torch.load(args.checkpoint_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    else:
        start_epoch = 0
    
    if args.aug_p_dict is None:
        args.aug_p_dict = {
                "collide": 0.4,
                "crop_shift": 0.4,
                "amp_jitter": 0.5,
                "temporal_jitter": 0.7,
                "smart_noise": (0.6, 1.0),
            }
    args.multi_chan = True if args.num_extra_chans > 0 else False
    
    ds = ContrastiveLearningDataset(
        args.data,
        args.out_dim,
        multi_chan=args.multi_chan,
    )
    train_dataset = ds.get_dataset(
        args.dataset_name,
        2,
        args.num_extra_chans,
        detected_spikes=args.detected_spikes,
        aug_p_dict=args.aug_p_dict
    )

    if args.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, drop_last=True
        )
        assert args.batch_size % args.world_size == 0
        per_device_batch_size = args.batch_size // args.world_size
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=per_device_batch_size,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
        )
    else:
        sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

    # define memory and test dataset for knn monitoring
    if args.rank == 0 and not args.no_eval:
        if args.multi_chan:
            memory_dataset = WFDataset_lab(
                args.data,
                split="train",
                multi_chan=args.multi_chan,
                transform=Crop(
                    prob=0.0, num_extra_chans=args.num_extra_chans, ignore_chan_num=True
                ),
                n_units=args.n_test_units,
                units_list=args.test_units_list,
            )
            memory_loader = torch.utils.data.DataLoader(
                memory_dataset,
                batch_size=128,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )
            test_dataset = WFDataset_lab(
                args.data,
                split="test",
                multi_chan=args.multi_chan,
                transform=Crop(
                    prob=0.0, num_extra_chans=args.num_extra_chans, ignore_chan_num=True
                ),
                n_units=args.n_test_units,
                units_list=args.test_units_list,
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            memory_dataset = WFDataset_lab(
                args.data,
                split="train",
                multi_chan=False,
                n_units=args.n_test_units,
                units_list=args.test_units_list,
            )
            memory_loader = torch.utils.data.DataLoader(
                memory_dataset,
                batch_size=128,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )
            test_dataset = WFDataset_lab(
                args.data,
                split="test",
                multi_chan=False,
                n_units=args.n_test_units,
                units_list=args.test_units_list,
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )
    else:
        memory_loader = None
        test_loader = None
        
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    noise_transform = TorchSmartNoise(
            args.data,
            noise_scale=args.aug_p_dict["smart_noise"][1],
            normalize=args.cell_type,
            gpu=gpu,
            p=args.aug_p_dict["smart_noise"][0],
        )
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.ddp:
            sampler.set_epoch(epoch)

        for step, (wf, chan_nums, lab) in enumerate(train_loader, start=epoch * len(train_loader)):
            wf = torch.cat(wf, dim=0).float()
            chan_nums = np.concatenate(chan_nums, axis=0)
            
            wf = wf.cuda(gpu, non_blocking=True)
            wf = noise_transform([wf, chan_nums])  # smart_noise on GPU

            if args.arch == 'scam':
                if not args.multi_chan:
                    wf = torch.squeeze(wf, dim=1)
                    wf = torch.unsqueeze(wf, dim=-1)
                else:
                    wf = wf.view(-1, (args.num_extra_chans * 2 + 1) * 121)
                    wf = torch.unsqueeze(wf, dim=-1)

            lr = args.learning_rate
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                y1, y2 = torch.chunk(wf, 2, dim=0)
                loss = model.forward(y1,y2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if args.rank == 0:
            # save checkpoint
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.checkpoint_dir / "checkpoint.pth")

            # save checkpoint to epoch
            if epoch % args.save_freq == 0 and epoch != 0:
                torch.save(
                    state, args.checkpoint_dir / "checkpoint_epoch{}.pth".format(epoch)
                )
                if not args.no_eval:
                    save_reps(
                        model,
                        memory_loader,
                        args.checkpoint_dir / "checkpoint.pth",
                        split="train",
                        multi_chan=True
                    )
                    save_reps(
                        model,
                        test_loader,
                        args.checkpoint_dir / "checkpoint.pth",
                        split="test",
                        multi_chan=True
                    )

            # log to tensorboard
            logger.log_value("loss", loss.item(), epoch)
            logger.log_value("learning_rate", lr, epoch)

            if epoch % args.eval_freq == 0 and not args.no_eval:
                knn_score = knn_monitor(
                    net=model,
                    memory_data_loader=memory_loader,
                    test_data_loader=test_loader,
                    device="cuda",
                    k=200,
                    hide_progress=True,
                    args=args,
                )
                gmm_score = gmm_monitor(
                    net=model,
                    memory_data_loader=memory_loader,
                    test_data_loader=test_loader,
                    device="cuda",
                    hide_progress=True,
                    args=args,
                )
                print(f"Epoch {epoch}, knn_acc:{knn_score}, gmm_acc:{gmm_score}")
                logger.log_value("knn_acc", knn_score, epoch)
                logger.log_value("gmm_acc", gmm_score, epoch)

    if args.rank == 0:
        # save final model
        torch.save(
            dict(
                backbone=model.module.backbone.state_dict(),
                projector=model.module.projector.state_dict(),
                head=model.module.online_head.state_dict(),
            ),
            args.checkpoint_dir / "final.pth",
        )



class SimCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.multi_chan = True if args.num_extra_chans > 0 else False
        
        if args.arch == 'scam':
            model_args = dict(
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                block_size=args.block_size,
                bias=args.bias,
                dropout=args.dropout,
                out_dim=args.out_dim,
                proj_dim=args.proj_dim,
                multi_chan=args.multi_chan,
                n_extra_chans=args.num_extra_chans,
                num_classes=args.num_classes,
            )
            scamconf = SCAMConfig(**model_args)
            self.backbone = Multi_SCAM(scamconf)
        else:
            self.backbone = ModelSimCLR(
                base_model=args.arch,
                out_dim=args.out_dim,
                proj_dim=args.proj_dim,
                fc_depth=args.fc_depth,
                expand_dim=args.expand_dim,
                multichan=args.multi_chan,
                input_size=(2 * args.num_extra_chans + 1) * 121,
            )
            if not args.eval_on_proj:
                self.backbone.backbone.proj = nn.Identity()
        self.args = args

        # projector
        if args.eval_on_proj:
            self.projector = None
        else:
            self.projector = Projector(rep_dim=args.out_dim, proj_dim=args.proj_dim)

    def forward(self, y1, y2=None):
        if y2 is None:
            return self.backbone(y1)
        z1 = self.backbone(y1)
        z2 = self.backbone(y2)
        
        # projection
        if self.projector is not None:
            z1 = self.projector(z1)
            z2 = self.projector(z2)
        z1, z2 = torch.squeeze(z1), torch.squeeze(z2)
        loss = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2
        
        return loss

def infoNCE(nn, p, temperature=0.2, gather_all=True):
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)
    if gather_all:
        nn = gather_from_all(nn)
        p = gather_from_all(p)
    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


def main(args):
    print("Starting Non-DDP training..")
    args.checkpoint_dir = args.checkpoint_dir / args.exp
    args.log_dir = args.log_dir / args.exp

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    main_worker(0, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Training")
    parser.add_argument(
        "--data",
        type=Path,
        metavar="DIR",
        default="./spike_data/dy016",
        help="path to dataset",
    )
    parser.add_argument(
        "-dataset-name",
        default="wfs",
        help="dataset name",
        choices=["wfs", "stl10", "cifar10"],
    )
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )
    parser.add_argument(
        "--epochs",
        default=800,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=128, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="base learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--save-freq", default=50, type=int, metavar="N", help="save frequency"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default="./saved_models/",
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default="./logs/",
        metavar="LOGDIR",
        help="path to tensorboard log directory",
    )
    parser.add_argument("--seed", default=42, type=int, help="seed")

    # Training / loss specific parameters
    parser.add_argument(
        "--temp", default=0.2, type=float, help="Temperature for InfoNCE loss"
    )


    # Slurm setting
    parser.add_argument(
        "--ngpus-per-node",
        default=6,
        type=int,
        metavar="N",
        help="number of gpus per node",
    )
    parser.add_argument(
        "--nodes", default=5, type=int, metavar="N", help="number of nodes"
    )
    parser.add_argument("--timeout", default=360, type=int, help="Duration of the job")
    parser.add_argument(
        "--partition", default="el8", type=str, help="Partition where to submit"
    )

    parser.add_argument("--exp", default="SimCLR", type=str, help="Name of experiment")

    # latent params
    parser.add_argument(
        "--out_dim", default=5, type=int, help="feature dimension (default: 5)"
    )
    parser.add_argument(
        "--proj_dim", default=5, type=int, help="projection dimension (default: 5)"
    )

    # MLP args
    parser.add_argument("--fc_depth", default=2, type=int)
    parser.add_argument("--expand_dim", default=16, type=int)

    # SCAM args
    parser.add_argument("--n_layer", default=20, type=int)
    parser.add_argument("--n_head", default=4, type=int)
    parser.add_argument("--n_embd", default=32, type=int)
    parser.add_argument(
        "--block_size", default=121, type=int
    )  # this is the max sequence length

    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--bias", action="store_true")  # default = False

    parser.add_argument("--online_head", action="store_true")  # default = False
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--num_extra_chans", default=0, type=int)
    parser.add_argument("--multi_chan", action="store_true")
    
    parser.add_argument("--eval-freq", default=1, type=int, metavar="N", help="eval frequency")
    parser.add_argument("--no_eval", action="store_true")  # default = False
    parser.add_argument("--eval_on_proj", action="store_true")  # default = False

    parser.add_argument("--cell_type", action="store_true")  # default = False

    parser.add_argument("--p_crop", default=0.5, type=float)
    parser.add_argument("--detected_spikes", action="store_true")  # default = False
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--aug_p_dict", default=None, nargs="+", type=float)  # prob of applying each aug in pipeline
    parser.add_argument(
        "--n_test_units", default=10, type=int
    )  # number of units to subsample for training metrics
    parser.add_argument(
        "--test_units_list", default=None, nargs="+", type=int
    )  # allows choosing the units for training metrics
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="scam",
        help="default: custom_encoder)",
        choices=["scam", "conv_encoder", "fc_encoder"],
    )
    args = parser.parse_args()

    main(args)
