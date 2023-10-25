import torch
import torch.nn as nn
from data_aug.contrastive_learning_dataset import (
    ContrastiveLearningDataset,
    WFDataset_lab,
)
from ceed.models.model_GPT import GPTConfig, Multi_GPT, Single_GPT, Projector
from data_aug.wf_data_augs import Crop
import os


# Loads single or multichannel Transformer backbone for analysis.
class Encoder(torch.nn.Module):
    def __init__(
        self,
        multi_chan=False,
        rep_dim=5,
        proj_dim=5,
        pos_enc="conseq",
        rep_after_proj=False,
        use_chan_pos=False,
        use_merge_layer=False,
        add_layernorm=False,
        half_embed_each=False,
        concat_pos=False,
        num_classes=10,
    ):
        super().__init__()
        if multi_chan:
            if concat_pos:
                model_args = dict(
                    bias=False,
                    block_size=1342,
                    n_layer=20,
                    n_head=4,
                    n_embd=64,
                    dropout=0.2,
                    out_dim=rep_dim,
                    proj_dim=proj_dim,
                    is_causal=True,
                    pos=pos_enc,
                    multi_chan=True,
                    use_chan_pos=use_chan_pos,
                    use_merge_layer=use_merge_layer,
                    add_layernorm=add_layernorm,
                    half_embed_each=half_embed_each,
                    concat_pos=concat_pos,
                    num_classes=num_classes,
                )
            else:
                model_args = dict(
                    bias=False,
                    block_size=1331,
                    n_layer=20,
                    n_head=4,
                    n_embd=64,
                    dropout=0.2,
                    out_dim=rep_dim,
                    proj_dim=proj_dim,
                    is_causal=True,
                    pos=pos_enc,
                    multi_chan=True,
                    use_chan_pos=use_chan_pos,
                    use_merge_layer=use_merge_layer,
                    add_layernorm=add_layernorm,
                    half_embed_each=half_embed_each,
                    concat_pos=concat_pos,
                    num_classes=num_classes,
                )
        else:
            model_args = dict(
                bias=False,
                block_size=121,
                n_layer=20,
                n_head=4,
                n_embd=32,
                dropout=0.0,
                out_dim=rep_dim,
                proj_dim=proj_dim,
                is_causal=True,
                pos=pos_enc,
                multi_chan=False,
                num_classes=num_classes,
            )
        gptconf = GPTConfig(**model_args)
        if multi_chan:
            self.backbone = Multi_GPT(gptconf)
        else:
            self.backbone = Single_GPT(gptconf)
            self.backbone.projector = (
                torch.nn.Identity()
            )  # this is loaded separately later
        if rep_after_proj:
            self.projector = Projector(
                rep_dim=gptconf.out_dim, proj_dim=gptconf.proj_dim
            )
        else:
            self.projector = None

    def forward(self, x, chan_pos=None):
        if chan_pos is not None:
            r = self.backbone(x, chan_pos=chan_pos)
        else:
            r = self.backbone(x)
        if self.projector is not None:
            r = self.projector(r)
        return r


# Loads GPT model from checkpoint
def load_ckpt(
    ckpt_path,
    multi_chan=False,
    rep_dim=5,
    proj_dim=5,
    pos_enc="conseq",
    rep_after_proj=False,
    use_chan_pos=False,
    use_merge_layer=False,
    add_layernorm=False,
    half_embed_each=False,
    concat_pos=False,
    num_classes=10,
):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model = Encoder(
        multi_chan=multi_chan,
        rep_dim=rep_dim,
        proj_dim=proj_dim,
        pos_enc=pos_enc,
        rep_after_proj=rep_after_proj,
        use_chan_pos=use_chan_pos,
        use_merge_layer=use_merge_layer,
        add_layernorm=add_layernorm,
        half_embed_each=half_embed_each,
        concat_pos=concat_pos,
        num_classes=num_classes,
    )
    if multi_chan:
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        m, uek = model.load_state_dict(state_dict, strict=False)
    else:
        state_dict = {
            "backbone." + k: v
            for k, v in ckpt["state_dict"].items()
            if "projector" not in k
        }
        state_dict.update(
            {k: v for k, v in ckpt["state_dict"].items() if "projector" in k}
        )
        m, uek = model.load_state_dict(state_dict, strict=False)
    print("missing keys", m)
    print("unexpected keys", uek)

    return model


# Loads GPT model from checkpoint
def load_ckpt_to_model(model, ckpt_path, multi_chan):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    if multi_chan:
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        m, uek = model.load_state_dict(state_dict, strict=False)
    else:
        state_dict = {
            k: v for k, v in ckpt["state_dict"].items() if "projector" not in k
        }
        state_dict.update(
            {k: v for k, v in ckpt["state_dict"].items() if "projector" in k}
        )
        m, uek = model.load_state_dict(state_dict, strict=False)
    print("missing keys", m)
    print("unexpected keys", uek)


# Loads GPT model from checkpoint
def load_GPT_backbone(backbone, checkpoint, is_multi_chan):
    if not is_multi_chan:
        state_dict = checkpoint["state_dict"]
        sd_keys = state_dict.keys()
        state_dict_backbone = {
            k: state_dict[k] for k in sd_keys if not k.endswith(".attn.bias")
        }
        backbone.load_state_dict(state_dict_backbone)
    else:
        state_dict = checkpoint["model"]
        sd_keys = state_dict.keys()
        state_dict_backbone = {
            k: state_dict[k]
            for k in sd_keys
            if k.startswith("module.backbone.") and not k.endswith(".attn.bias")
        }
        backbone_keys = state_dict_backbone.keys()
        state_dict_backbone_final = {
            k.replace("module.", "").replace("backbone.", ""): state_dict_backbone[k]
            for k in backbone_keys
        }
        backbone.load_state_dict(state_dict_backbone_final)
    return backbone


# gets encoder backbone from fully connected and convolutional encoders
def get_backbone(enc):
    last_layer = list(list(enc.children())[-1].children())[:-1]
    enc.fcpart = nn.Sequential(*last_layer)
    return enc


# returns waveform data loader from a dataset of spikes
def get_dataloader(data_path, multi_chan=False, split="train", use_chan_pos=False):
    if multi_chan:
        dataset = WFDataset_lab(
            data_path,
            split=split,
            multi_chan=True,
            use_chan_pos=use_chan_pos,
            transform=Crop(prob=0.0, num_extra_chans=5, ignore_chan_num=True),
        )
    else:
        print("Loading single channel data")
        dataset = WFDataset_lab(data_path, split=split, multi_chan=False)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    return loader
