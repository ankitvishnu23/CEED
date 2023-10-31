from pathlib import Path
import logging
import os
import uuid
import subprocess

import submitit
import numpy as np
import argparse

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
_logger = logging.getLogger('train')

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
parser.add_argument("--ddp", action="store_true", default=True)
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
    
# Slurm setting
parser.add_argument('--ngpus-per-node', default=6, type=int, metavar='N',
                    help='number of gpus per node')
parser.add_argument('--nodes', default=5, type=int, metavar='N',
                    help='number of nodes')
parser.add_argument("--timeout", default=360, type=int,
                    help="Duration of the job")
parser.add_argument("--partition", default="xxx", type=str,
                    help="Partition where to submit")

parser.add_argument("--exp", default="expt_name", type=str,
                    help="Name of experiment")

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main
        self._setup_gpu_args()
        main.main_worker(self.args.gpu, self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(self.args).as_uri()
        checkpoint_file = os.path.join(self.args.checkpoint_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(args.job_dir, exist_ok=True)
    init_file = args.job_dir / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def main():
    args = parser.parse_args()

    args.checkpoint_dir = args.checkpoint_dir / args.exp
    args.log_dir = args.log_dir / args.exp
    args.job_dir = args.checkpoint_dir

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    get_init_file(args)

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus_per_node
    nodes = args.nodes
    timeout_min = args.timeout
    partition = args.partition

    kwargs = {'slurm_gres': f'gpu:{num_gpus_per_node}',}

    executor.update_parameters(
        mem_gb=30 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=24,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 6
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.exp)

    args.dist_url = get_init_file(args).as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    _logger.info("Submitted job_id:", job.job_id)


if __name__ == '__main__':
    main()
