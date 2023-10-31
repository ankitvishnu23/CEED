import os
import shutil

import torch
import torch.nn as nn
import numpy as np
import shutil
import random

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """Saves checkpoint of the model"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def save_config_file(model_checkpoints_folder, args):
    """Saves config args to a folder as a config file"""
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, "config.yml"), "w") as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def get_contr_representations(model, data_set, device="cpu"):
    """Computes representations of data using a model. Will not work with multichan or Transformer encoders."""
    reps = []
    for item in data_set:
        with torch.no_grad():
            wf = torch.from_numpy(item.reshape(1, 1, -1)).to(device)
            rep = model(wf)
        reps.append(rep.cpu().numpy())
    return np.squeeze(np.array(reps))


def get_torch_reps(net, data_loader, device, args):
    """Computes representations of waveforms and returns them with the corresponding labels
    Args:
        net:
            encoder network to use for extracting representations.
        data_loader: torch.utils.DataLoader
            data loader built from a WF Dataset.
        device: str
            'cuda' or 'cpu' device.
        args: Namespace
            configurations for model loading.
    """
    feature_bank = []
    feature_labels = torch.tensor([])
    with torch.no_grad():
        # generate feature bank
        for data, _, target in data_loader:

            data = data.float()

            if args.arch == 'scam':
                data = (
                    data.view(-1, (args.num_extra_chans * 2 + 1) * 121)
                    if args.multi_chan
                    else torch.squeeze(data, dim=1)
                )
                
                feature = net(
                    data.to(device=device, non_blocking=True).unsqueeze(dim=-1)
                )
            else:
                feature = net(data.to(device=device, non_blocking=True))
                feature = torch.squeeze(feature)
            if len(feature.shape) > 1:
                dim = 1
            else:
                dim = 0
            feature = F.normalize(feature, dim=dim)
            feature_bank.append(feature)
            feature_labels = torch.cat((feature_labels, target))
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).cpu().numpy()
        # [N]
        feature_labels = feature_labels.cpu().numpy()

    return feature_bank, feature_labels


def get_torch_reps_nolabels(net, data_loader, device, args):
    """Computes representations of waveforms and returns them
    Args:
        net:
            encoder network to use for extracting representations.
        data_loader: torch.utils.DataLoader
            data loader built from a WF Dataset.
        device: str
            'cuda' or 'cpu' device.
        args: Namespace
            configurations for model loading.
    """
    feature_bank = []
    with torch.no_grad():
        # generate feature bank
        for data in data_loader:
            data = data.to(dtype=torch.float32)
            if args.arch == 'scam':
                data = (
                    data.view(-1, (args.num_extra_chans * 2 + 1) * 121)
                    if args.multi_chan
                    else torch.squeeze(data, dim=1)
                )
                feature = net(
                    data.to(device=device, non_blocking=True).unsqueeze(dim=-1)
                )
            else:
                feature = net(data.to(device=device, non_blocking=True))
                feature = torch.squeeze(feature)
            if len(feature.shape) > 1:
                dim = 1
            else:
                dim = 0
            feature = F.normalize(feature, dim=dim)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).cpu().numpy()

    return feature_bank


def apply_transform(transform, data):
    """Applies transformation specified to wfs
    Args:
        transform: object
            transform defined in wf_data_augs.
        data:
            wfs to transform.
    """
    transformed_data = np.array([transform(d) for d in data])
    return transformed_data


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, classes, device=sim_labels.device
    )
    # [B*K, C]
    assert (sim_labels >= 0).all(), "label indices less than 0 detected!"
    assert (sim_labels < classes).all(), "label indices grater than classes detected!"
    assert one_hot_label.device == sim_labels.device, "Device mismatch detected!"
    assert sim_labels.dtype == torch.int64, "sim_labels should be of dtype torch.int64!"

    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


# GMM fitting to data for validation
def gmm_monitor(
    net,
    memory_data_loader,
    test_data_loader,
    device="cuda",
    hide_progress=False,
    epoch_num=0,
    targets=None,
    args=None,
):
    if not targets:
        targets = memory_data_loader.dataset.targets

    net.eval()

    num_iters = 50 if epoch_num == args.epochs - 1 else 1

    scores = []
    np.random.seed(0)
    random.seed(0)
    reps_train, labels_train = get_torch_reps(net, memory_data_loader, device, args)
    reps_test, labels_test = get_torch_reps(net, test_data_loader, device, args)
    classes = max(len(np.unique(labels_train)), len(np.unique(labels_test)))
    for i in range(num_iters):
        # covariance_type : {'full', 'tied', 'diag', 'spherical'}
        covariance_type = "full"
        gmm = GaussianMixture(
            classes,
            random_state=random.randint(0, 1000000),
            covariance_type=covariance_type,
        ).fit(reps_train)
        gmm_cont_test_labels = gmm.predict(reps_test)
        curr_score = adjusted_rand_score(labels_test, gmm_cont_test_labels) * 100
        scores.append(curr_score)
        if i == 49:
            print("max gmm score: {}".format(max(scores)))
            print("min gmm score: {}".format(min(scores)))
            print("50 run gmm mean score: {}".format(np.mean(scores)))
            print("50 run gmm std-dev score: {}".format(np.std(scores)))

    score = np.mean(scores)

    return score


# test using a knn monitor
def knn_monitor(
    net,
    memory_data_loader,
    test_data_loader,
    device="cuda",
    k=200,
    t=0.1,
    hide_progress=False,
    targets=None,
    args=None,
):
    if not targets:
        targets = memory_data_loader.dataset.targets

    net.eval()
    classes = len(np.unique(targets))
    # classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in memory_data_loader:
            if not args.multi_chan:
                if args.arch == 'scam':
                    data = torch.squeeze(data, dim=1)
                    data = torch.unsqueeze(data, dim=-1)
            else:
                data = data.view(-1, int(args.num_extra_chans * 2 + 1) * 121)
                data = torch.unsqueeze(data, dim=-1)
            data = data.float()
            feature = net(data.to(device=device, non_blocking=True))
            
            feature = torch.squeeze(feature)
            
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device)

        # loop test data to predict the label by weighted knn search
        for data, _, target in test_data_loader:
            target = target.to(device=device, non_blocking=True)
            if not args.multi_chan:
                data = torch.squeeze(data, dim=1)
                data = torch.unsqueeze(data, dim=-1)
            else:
                data = data.view(-1, int(args.num_extra_chans * 2 + 1) * 121)
                data = torch.unsqueeze(data, dim=-1)
            data = data.float()

            feature = net(data.to(device=device, non_blocking=True))
            
            feature = torch.squeeze(feature)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(
                feature, feature_bank, feature_labels, classes, k, t
            )

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
    return total_top1 / total_num * 100


def save_reps(
    model,
    loader,
    ckpt_path,
    split="train",
    multi_chan=False,
    suffix="",
):
    ckpt_path=str(ckpt_path)
    ckpt_root_dir = "/".join(ckpt_path.split("/")[:-1])
    model.eval()
    feature_bank = []
    with torch.no_grad():
        for data, _, target in loader:
            if not multi_chan:
                data = torch.squeeze(data, dim=1)
                data = torch.unsqueeze(data, dim=-1)
            else:
                data = data.view(-1, 11 * 121)
                data = torch.unsqueeze(data, dim=-1)
            data = data.float()

            feature = model(data.cuda(non_blocking=True))
            feature_bank.append(feature)

        feature_bank = torch.cat(feature_bank, dim=0)
        torch.save(
            feature_bank, os.path.join(ckpt_root_dir, f"{split}_reps{suffix}.pt")
        )
        print(f"saved {split} features to {ckpt_root_dir}/{split}_reps{suffix}.pt")
