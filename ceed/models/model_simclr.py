import math
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from ceed.models.exceptions import InvalidBackboneError
from collections import OrderedDict
import sys


class Projector(nn.Module):
    """Projector network accepts a variable number of layers indicated by depth.
    Option to include batchnorm after every layer."""

    def __init__(self, Lvpj=[512, 128], rep_dim=5, proj_dim=5, bnorm=False, depth=3):
        super().__init__()
        print(
            f"Using projector; batchnorm {bnorm} with depth {depth}; hidden_dim={Lvpj[0]}"
        )
        nlayer = [nn.BatchNorm1d(Lvpj[0])] if bnorm else []
        list_layers = [nn.Linear(rep_dim, Lvpj[0])] + nlayer + [nn.ReLU()]
        for _ in range(depth - 2):
            list_layers += [nn.Linear(Lvpj[0], Lvpj[0])] + nlayer + [nn.ReLU()]
        list_layers += [nn.Linear(Lvpj[0], proj_dim)]
        self.proj_block = nn.Sequential(*list_layers)

    def forward(self, x):
        x = self.proj_block(x)
        return x


class FullyConnectedEnc(nn.Module):
    def __init__(
        self,
        input_size=121,
        Lv=[768, 512, 256],
        out_size=2,
        proj_dim=5,
        fc_depth=2,
        multichan=False,
    ):
        super().__init__()
        #   self.proj_dim = out_size if out_size < proj_dim else proj_dim
        self.proj_dim = proj_dim
        self.input_size = input_size
        self.multichan = multichan

        self.fcpart = nn.Sequential(
            nn.Linear(input_size, Lv[0]),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(Lv[0], Lv[1]),
            nn.ReLU(),
            nn.Linear(Lv[1], Lv[2]),
            nn.ReLU(),
            nn.Linear(Lv[2], out_size),
            Projector(rep_dim=out_size, proj_dim=self.proj_dim),
        )
        self.Lv = Lv

    def forward(self, x):
        if self.multichan:
            x = x.view(-1, 1, self.input_size)
        x = self.fcpart(x)
        return x

    def load(self, fname_model):
        checkpoint = torch.load(fname_model, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for key in state_dict:
            # if "backbone" in key and "fc" not in key:
            new_key = ".".join(key.split(".")[1:])
            new_state_dict[new_key] = state_dict[key]
        self.load_state_dict(new_state_dict)
        return self


class Encoder(nn.Module):
    def __init__(
        self,
        Lv=[200, 150, 100, 75],
        ks=[11, 21, 31],
        out_size=2,
        proj_dim=5,
        fc_depth=2,
        input_size=121,
    ):
        super().__init__()
        print("init Encoder")
        self.proj_dim = out_size if out_size < proj_dim else proj_dim
        self.enc_block1d = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=Lv[0],
                kernel_size=ks[0],
                padding=math.ceil((ks[0] - 1) / 2),
            ),
            nn.BatchNorm1d(Lv[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(p=0.2),
            nn.Conv1d(Lv[0], Lv[1], ks[1], padding=math.ceil((ks[1] - 1) / 2)),
            nn.BatchNorm1d(Lv[1]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            # nn.Dropout(p=0.2),
            nn.Conv1d(Lv[1], Lv[2], ks[2], padding=math.ceil((ks[2] - 1) / 2)),
            nn.BatchNorm1d(Lv[2]),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.avgpool1d = nn.AdaptiveAvgPool1d((1))

        self.fcpart = nn.Sequential(
            nn.Linear(Lv[2] * 1 * 1, Lv[3]),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(Lv[3], out_size),
            Projector(rep_dim=out_size, proj_dim=self.proj_dim),
        )
        self.Lv = Lv

    def forward(self, x):
        x = self.enc_block1d(x)
        x = self.avgpool1d(x)
        x = x.view(-1, self.Lv[2] * 1 * 1)
        x = self.fcpart(x)
        return x

    def load(self, fname_model):
        checkpoint = torch.load(fname_model, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for key in state_dict:
            # if "backbone" in key and "fc" not in key:
            new_key = ".".join(key.split(".")[1:])
            new_state_dict[new_key] = state_dict[key]
        self.load_state_dict(new_state_dict)
        return self


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        num_chans: int = 1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.num_chans = num_chans

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        sin_arr = torch.sin(position * div_term)
        cos_arr = torch.cos(position * div_term)
        if self.num_chans > 1:
            pe = torch.zeros(1, self.num_chans, max_len, d_model)
            for i in range(self.num_chans):
                pe[0, i, :, 0::2] = sin_arr
                pe[0, i, :, 1::2] = cos_arr
        else:
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = sin_arr
            pe[0, :, 1::2] = cos_arr

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if self.num_chans > 1:
            x = x + self.pe[:, :, : x.size(2)]
        else:
            x = x + self.pe[:, :, : x.size(1)]

        return self.dropout(x)


class MultiChanAttentionEnc(nn.Module):
    def __init__(
        self,
        spike_size=121,
        n_channels=11,
        out_size=2,
        proj_dim=5,
        fc_depth=2,
        nlayers=9,
        nhead=8,
        dropout=0.1,
        expand_dim=16,
        cls_head=None,
    ):
        super().__init__()
        self.spike_size = spike_size
        self.expand_dim = expand_dim
        self.n_channels = n_channels
        self.proj_dim = out_size if out_size < proj_dim else proj_dim
        if expand_dim != 1:
            self.encoder = nn.Linear(1, expand_dim)
        else:
            nhead = 1
        self.pos_encoder = PositionalEncoding(
            expand_dim, dropout, spike_size, num_chans=n_channels
        )
        encoder_layers = TransformerEncoderLayer(expand_dim, nhead, 512)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder_sum = nn.Linear(n_channels, 1)
        list_layers = [
            nn.Linear(self.spike_size * self.expand_dim, 256),
            nn.ReLU(inplace=True),
        ]
        for _ in range(fc_depth - 2):
            list_layers += [nn.Linear(256, 256), nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(256, out_size)]
        if not cls_head:
            list_layers += [Projector(rep_dim=out_size, proj_dim=self.proj_dim)]
        else:
            print(f"using head = {cls_head}")
            if cls_head == "linear":
                self.cls_head = nn.Linear(out_size, 10)
            elif cls_head == "mlp2":
                self.cls_head = nn.Sequential(
                    nn.Linear(out_size, 100), nn.ReLU(inplace=True), nn.Linear(100, 10)
                )
            elif cls_head == "mlp3":
                self.cls_head = nn.Sequential(
                    nn.Linear(out_size, 100),
                    nn.ReLU(inplace=True),
                    nn.Linear(100, 50),
                    nn.ReLU(inplace=True),
                    nn.Linear(50, 10),
                )
            list_layers += [self.cls_head]

        self.fcpart = nn.Sequential(*list_layers)

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.fcpart.bias.data.zero_()
        self.fpcart.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [batch_size, proj_dim]
        """

        if self.expand_dim != 1:
            src = torch.unsqueeze(src, dim=-1)
        src = self.pos_encoder(src)  # [B, n_chans, seq_len, embed_dim]
        src = src.reshape(
            -1, self.spike_size, self.expand_dim
        )  # [B * n_chans, seq_len, embed_dim]
        src = src.permute(
            1, 0, 2
        )  # remove batch_first argument needs Batch in second dim

        output = self.transformer_encoder(src, src_mask)
        output = output.permute(1, 0, 2)
        output = output.reshape(-1, self.n_channels, self.spike_size * self.expand_dim)
        output = torch.transpose(output, 1, 2)
        output = self.encoder_sum(output)
        output = torch.squeeze(output)

        output = self.fcpart(output)
        return output

    def load(self, fname_model):
        checkpoint = torch.load(fname_model, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for key in state_dict:
            # if "backbone" in key and "fc" not in key:
            new_key = ".".join(key.split(".")[1:])
            new_state_dict[new_key] = state_dict[key]
            if "pos_encoder" in key:
                new_state_dict[new_key] = state_dict[key].transpose(0, 1)
        self.load_state_dict(new_state_dict)
        return self


model_dict = {
    "fc_encoder": FullyConnectedEnc,
    "conv_encoder": Encoder,
    "attention_encoder": MultiChanAttentionEnc,
}


class ModelSimCLR(nn.Module):
    def __init__(
        self,
        base_model,
        out_dim,
        proj_dim,
        fc_depth=2,
        expand_dim=16,
        ckpt=None,
        cls_head=None,
        multichan=True,
        input_size=121,
    ):
        super().__init__()
        if "attention" in base_model:
            self.backbone = model_dict[base_model](
                out_size=out_dim,
                proj_dim=proj_dim,
                fc_depth=fc_depth,
                expand_dim=expand_dim,
                cls_head=cls_head,
            )
        else:
            self.backbone = model_dict[base_model](
                out_size=out_dim,
                proj_dim=proj_dim,
                fc_depth=fc_depth,
                input_size=input_size,
                multichan=multichan,
            )

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
        except KeyError:
            raise InvalidBackboneError("Invalid backbone architecture")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
