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

from utils.ddp_utils import gather_from_all

from data_aug.contrastive_learning_dataset import (
    ContrastiveLearningDataset,
    WFDataset_lab,
)
from ceed.models.model_GPT import GPTConfig, Multi_GPT, Projector
from utils.utils import knn_monitor, gmm_monitor, save_reps
from data_aug.wf_data_augs import Crop

# def main():
#     args = parser.parse_args()
#     args.ngpus_per_node = torch.cuda.device_count()
#     args.scale = [float(x) for x in args.scale.split(',')]
#     if 'SLURM_JOB_ID' in os.environ:
#         cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
#         stdout = subprocess.check_output(cmd.split())
#         host_name = stdout.decode().splitlines()[0]
#         args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
#         args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
#         args.dist_url = f'tcp://{host_name}:58478'
#     else:
#         # single-node distributed training
#         args.rank = 0
#         args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
#         args.world_size = args.ngpus_per_node
#     torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


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

    if args.optimizer == "lars":
        optimizer = LARS(
            model.parameters(),
            lr=0,
            weight_decay=args.weight_decay,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm,
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0,
            momentum=args.opt_momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), args.learning_rate, weight_decay=args.weight_decay
        )

    # build memory bank and its loss
    mem_bank = None

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / "checkpoint.pth").is_file():
        ckpt = torch.load(args.checkpoint_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    else:
        start_epoch = 0

    num_extra_chans = args.num_extra_chans if args.multi_chan else 0
    ds = ContrastiveLearningDataset(
        args.data,
        args.out_dim,
        multi_chan=args.multi_chan,
        use_chan_pos=args.use_chan_pos,
    )
    dataset = ds.get_dataset(
        "wfs",
        2,
        args.noise_scale,
        num_extra_chans,
        normalize=args.cell_type,
        p_crop=args.p_crop,
        detected_spikes=args.detected_spikes,
    )

    if args.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, drop_last=True
        )
        assert args.batch_size % args.world_size == 0
        per_device_batch_size = args.batch_size // args.world_size
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
        )
    else:
        sampler = None
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

    if args.rank == 0 and not args.no_knn:
        if args.multi_chan:
            memory_dataset = WFDataset_lab(
                args.data,
                split="train",
                multi_chan=args.multi_chan,
                transform=Crop(
                    prob=0.0, num_extra_chans=num_extra_chans, ignore_chan_num=True
                ),
                use_chan_pos=args.use_chan_pos,
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
                    prob=0.0, num_extra_chans=num_extra_chans, ignore_chan_num=True
                ),
                use_chan_pos=args.use_chan_pos,
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
            memory_dataset = WFDataset_lab(args.data, split="train", multi_chan=False)
            memory_loader = torch.utils.data.DataLoader(
                memory_dataset,
                batch_size=128,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )
            test_dataset = WFDataset_lab(args.data, split="test", multi_chan=False)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    # test knn first
    # if args.rank == 0:
    # knn_score = knn_monitor(net=model, memory_data_loader=memory_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
    # print(f"my knn_acc:{knn_score}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.ddp:
            sampler.set_epoch(epoch)

        for step, (wf, labels) in enumerate(loader, start=epoch * len(loader)):
            labels = labels[0].long()

            if args.use_chan_pos:
                y1 = wf[0][0].float()
                y2 = wf[1][0].float()
                chan_pos = wf[0][1].float()
                chan_pos2 = wf[1][1].float()

            else:
                y1 = wf[0].float()
                y2 = wf[1].float()
                chan_pos = None
                chan_pos2 = None

            if not args.multi_chan:
                y1, y2 = torch.squeeze(y1, dim=1), torch.squeeze(y2, dim=1)
                y1, y2 = torch.unsqueeze(y1, dim=-1), torch.unsqueeze(y2, dim=-1)
            else:
                y1, y2 = y1.view(-1, (args.num_extra_chans * 2 + 1) * 121), y2.view(
                    -1, (args.num_extra_chans * 2 + 1) * 121
                )
                y1, y2 = torch.unsqueeze(y1, dim=-1), torch.unsqueeze(y2, dim=-1)
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            if args.use_chan_pos:
                chan_pos = chan_pos.cuda(gpu, non_blocking=True)
                chan_pos2 = chan_pos2.cuda(gpu, non_blocking=True)

            labels = labels.cuda(gpu, non_blocking=True)
            if args.optimizer != "adam":
                lr = adjust_learning_rate(args, optimizer, loader, step)
            else:
                lr = args.learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss, acc = model.forward(
                    y1, y2, labels, chan_pos=chan_pos, chan_pos2=chan_pos2
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % args.print_freq == 0:
                if args.ddp and acc.item() >= 0:
                    torch.distributed.reduce(acc.div_(args.world_size), 0)
                if args.rank == 0:
                    print(
                        f"epoch={epoch}, step={step}, loss={loss.item()}, acc={acc.item()}, time={int(time.time() - start_time)}",
                        flush=True,
                    )
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        learning_rate=lr,
                        loss=loss.item(),
                        acc=acc.item(),
                        time=int(time.time() - start_time),
                    )
                    print(json.dumps(stats), file=stats_file)

        if args.rank == 0:
            # save checkpoint
            if args.memory_bank:
                state = dict(
                    epoch=epoch + 1,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    mem_bank=mem_bank.state_dict(),
                )
            else:
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

            if epoch % 50 == 0 and not args.no_knn:
                save_reps(
                    model,
                    memory_loader,
                    args.checkpoint_dir / "checkpoint.pth",
                    split="train",
                    multi_chan=True,
                    rep_after_proj=False,
                    use_chan_pos=args.use_chan_pos,
                )
                save_reps(
                    model,
                    test_loader,
                    args.checkpoint_dir / "checkpoint.pth",
                    split="test",
                    multi_chan=True,
                    rep_after_proj=False,
                    use_chan_pos=args.use_chan_pos,
                )

            # log to tensorboard
            logger.log_value("loss", loss.item(), epoch)
            logger.log_value("acc", acc.item(), epoch)
            logger.log_value("learning_rate", lr, epoch)

            if epoch % args.knn_freq == 0 and not args.no_knn:
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
                print(f"Epoch {epoch}, my knn_acc:{knn_score}")
                print(f"Epoch {epoch}, my gmm_acc:{gmm_score}")
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
            args.checkpoint_dir / "resnet50.pth",
        )


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.learning_rate  # * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class SimCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_extra_chans = args.num_extra_chans if args.multi_chan else 0
        model_args = dict(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            block_size=args.block_size,
            bias=args.bias,
            vocab_size=args.vocab_size,
            dropout=args.dropout,
            out_dim=args.out_dim,
            is_causal=args.is_causal,
            proj_dim=args.proj_dim,
            pos=args.pos_enc,
            multi_chan=args.multi_chan,
            use_chan_pos=args.use_chan_pos,
            n_extra_chans=num_extra_chans,
            add_layernorm=args.add_layernorm,
            use_merge_layer=args.use_merge_layer,
            half_embed_each=args.half_embed_each,
            remove_pos=args.remove_pos,
            concat_pos=args.concat_pos,
            num_classes=args.num_classes,
        )
        gptconf = GPTConfig(**model_args)
        self.backbone = Multi_GPT(gptconf)
        self.args = args

        # projector
        self.projector = Projector(rep_dim=gptconf.out_dim, proj_dim=gptconf.proj_dim)
        self.online_head = nn.Linear(gptconf.out_dim, gptconf.num_classes)  # 10 classes

    def forward(self, y1, y2=None, labels=None, chan_pos=None, chan_pos2=None):
        if y2 is None:
            if chan_pos is not None:
                r1 = self.backbone(y1, chan_pos=chan_pos)
            else:
                r1 = self.backbone(y1)
            # z1 = self.projector(r1)
            return r1
        if chan_pos is not None:
            r1 = self.backbone(y1, chan_pos=chan_pos)
            r2 = self.backbone(y2, chan_pos=chan_pos2)
        else:
            r1 = self.backbone(y1)
            r2 = self.backbone(y2)

        # projoection
        z1 = self.projector(r1)
        z2 = self.projector(r2)

        loss = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2

        logits = self.online_head(r1.detach())
        if not self.args.detected_spikes:
            cls_loss = torch.nn.functional.cross_entropy(logits, labels)
            acc = torch.sum(
                torch.eq(torch.argmax(logits, dim=1), labels)
            ) / logits.size(0)
            loss = loss + cls_loss
        else:
            acc = torch.Tensor([-1.0])

        return loss, acc


def build_loss_fn(args):
    return infoNCE


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


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def exclude_bias_and_norm(p):
    return p.ndim == 1


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
        default="/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_data/dy016",
        help="path to dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "cifar100"],
        help="dataset (imagenet, cifar100)",
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
        "--print-freq", default=10, type=int, metavar="N", help="print frequency"
    )
    parser.add_argument(
        "--save-freq", default=10, type=int, metavar="N", help="save frequency"
    )
    parser.add_argument(
        "--topk-path",
        type=str,
        default="./imagenet_resnet50_top10.pkl",
        help="path to topk predictions from pre-trained classifier",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default="/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_ddp/saved_models_int/",
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default="/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_ddp/logs_int/",
        metavar="LOGDIR",
        help="path to tensorboard log directory",
    )
    parser.add_argument(
        "--rotation", default=0.0, type=float, help="coefficient of rotation loss"
    )
    parser.add_argument("--scale", default="0.05,0.14", type=str)
    parser.add_argument("--seed", default=42, type=int, help="seed")

    # Training / loss specific parameters
    parser.add_argument(
        "--temp", default=0.2, type=float, help="Temperature for InfoNCE loss"
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="",
        help="Masking mode (masking out only positives, masking out all others than the topk classes",
        choices=[
            "pos",
            "supcon",
            "supcon_all",
            "topk",
            "topk_sum",
            "topk_agg_sum",
            "weight_anchor_logits",
            "weight_class_logits",
        ],
    )
    parser.add_argument(
        "--topk", default=5, type=int, metavar="K", help="Top k classes to use"
    )
    parser.add_argument(
        "--topk-only-first",
        action="store_true",
        default=False,
        help="Whether to only use the first block of anchors",
    )
    parser.add_argument(
        "--memory-bank",
        action="store_true",
        default=False,
        help="Whether to use memory bank",
    )
    parser.add_argument(
        "--mem-size", default=100000, type=int, help="Size of memory bank"
    )
    parser.add_argument(
        "--opt-momentum", default=0.9, type=float, help="Momentum for optimizer"
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        type=str,
        help="Optimizer",
        choices=["lars", "sgd", "adam"],
    )

    # Transform
    parser.add_argument(
        "--weak-aug",
        action="store_true",
        default=False,
        help="Whether to use augmentation reguarlization (strong & weak augmentation)",
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

    # new params
    parser.add_argument(
        "--out_dim", default=5, type=int, help="feature dimension (default: 5)"
    )
    parser.add_argument(
        "--proj_dim", default=5, type=int, help="projection dimension (default: 5)"
    )
    parser.add_argument("--multi_chan", default=False, action="store_true")
    parser.add_argument("--pos_enc", default="seq_11times", type=str)

    parser.add_argument("--use_gpt", action="store_true")  # default = False
    parser.add_argument(
        "-ns",
        "--noise_scale",
        default=1.0,
        help="how much to scale the noise augmentation (default: 1)",
    )

    # GPT args
    parser.add_argument("--n_layer", default=20, type=int)
    parser.add_argument("--n_head", default=4, type=int)
    parser.add_argument("--n_embd", default=32, type=int)
    parser.add_argument("--is_causal", action="store_true")  # default = False
    # parser.add_argument('--block_size', default=2678, type=int) # this is the max sequence length
    parser.add_argument(
        "--block_size", default=121, type=int
    )  # this is the max sequence length

    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--bias", action="store_true")  # default = False
    parser.add_argument(
        "--vocab_size", default=50304, type=int
    )  # default to GPT-2 vocab size
    parser.add_argument("--online_head", action="store_true")  # default = False
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--num_extra_chans", default=0, type=int)
    parser.add_argument(
        "--knn-freq", default=1, type=int, metavar="N", help="save frequency"
    )
    parser.add_argument("--add_train", action="store_true")  # default = False
    parser.add_argument("--use_chan_pos", action="store_true")  # default = False
    parser.add_argument("--use_merge_layer", action="store_true")  # default = False
    parser.add_argument("--add_layernorm", action="store_true")  # default = False
    parser.add_argument("--cell_type", action="store_true")  # default = False
    parser.add_argument("--no_knn", action="store_true")  # default = False

    parser.add_argument("--half_embed_each", action="store_true")  # default = False
    parser.add_argument("--remove_pos", action="store_true")  # default = False
    parser.add_argument("--p_crop", default=0.5, type=float)
    parser.add_argument("--detected_spikes", action="store_true")  # default = False
    parser.add_argument("--concat_pos", action="store_true")  # default = False
    parser.add_argument("--num_classes", default=10, type=int)

    args = parser.parse_args()

    main(args)
