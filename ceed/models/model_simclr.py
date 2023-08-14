import math
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from ceed.models.exceptions import InvalidBackboneError
from collections import OrderedDict
import sys

class SingleChanDenoiser(nn.Module):
    """Cleaned up a little. Why is conv3 here and commented out in forward?"""

    def __init__(
        # self, n_filters=[16, 8, 4], filter_sizes=[5, 11, 21], spike_size=121
        self, n_filters=[16, 8], filter_sizes=[5, 11], spike_size=121
    ):
        super(SingleChanDenoiser, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, n_filters[0], filter_sizes[0]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(n_filters[0], n_filters[1], filter_sizes[1]), nn.ReLU())
        if len(n_filters) > 2:
            self.conv3 = nn.Sequential(nn.Conv1d(n_filters[1], n_filters[2], filter_sizes[2]), nn.ReLU())
        n_input_feat = n_filters[1] * (spike_size - filter_sizes[0] - filter_sizes[1] + 2)
        self.out = nn.Linear(n_input_feat, spike_size)

    def forward(self, x):
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        return self.out(x)

    def load(self, fname_model):
        checkpoint = torch.load(fname_model, map_location="cpu")
        self.load_state_dict(checkpoint, strict=False)
        return self


class Projector(nn.Module):
    ''' Projector network accepts a variable number of layers indicated by depth.
    Option to include batchnorm after every layer.'''

    def __init__(self, Lvpj=[512, 128], rep_dim=5, proj_dim=5, bnorm = False, depth = 3):
        super(Projector, self).__init__()
        print(f"Using projector; batchnorm {bnorm} with depth {depth}; hidden_dim={Lvpj[0]}")
        nlayer = [nn.BatchNorm1d(Lvpj[0])] if bnorm else []
        list_layers = [nn.Linear(rep_dim, Lvpj[0])] + nlayer + [nn.ReLU()]
        for _ in range(depth-2):
            list_layers += [nn.Linear(Lvpj[0], Lvpj[0])] + nlayer + [nn.ReLU()]
        list_layers += [nn.Linear(Lvpj[0], proj_dim)]
        self.proj_block = nn.Sequential(*list_layers)

    def forward(self, x):
        x = self.proj_block(x)
        return x


class Projector2(nn.Module):
    ''' Projector network accepts a variable number of layers indicated by depth.
    Option to include batchnorm after every layer.'''

    def __init__(self, Lvpj=[128], rep_dim=5, proj_dim=5, bnorm = False, depth = 2):
        super(Projector2, self).__init__()
        print(f"Using projector; batchnorm {bnorm} with depth {depth}; hidden_dim={Lvpj[0]}")
        nlayer = [nn.BatchNorm1d(Lvpj[0])] if bnorm else []
        list_layers = [nn.Linear(rep_dim, Lvpj[0])] + nlayer + [nn.ReLU()]
        for _ in range(depth-2):
            list_layers += [nn.Linear(Lvpj[0], Lvpj[0])] + nlayer + [nn.ReLU()]
        list_layers += [nn.Linear(Lvpj[0], proj_dim)]
        self.proj_block = nn.Sequential(*list_layers)

    def forward(self, x):
        x = self.proj_block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, Lv=[200, 150, 100, 75], ks=[11, 21, 31], out_size = 2, proj_dim=5, fc_depth=2, input_size=121):
        super(Encoder, self).__init__()
        print("init Encoder")
        self.proj_dim = out_size if out_size < proj_dim else proj_dim
        self.enc_block1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=Lv[0], kernel_size=ks[0], padding=math.ceil((ks[0]-1)/2)),
            nn.BatchNorm1d(Lv[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(p=0.2),
            nn.Conv1d(Lv[0], Lv[1], ks[1], padding=math.ceil((ks[1]-1)/2)),
            nn.BatchNorm1d(Lv[1]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            # nn.Dropout(p=0.2),
            nn.Conv1d(Lv[1], Lv[2], ks[2], padding=math.ceil((ks[2]-1)/2)),
            nn.BatchNorm1d(Lv[2]),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        self.avgpool1d = nn.AdaptiveAvgPool1d((1))

        self.fcpart = nn.Sequential(
            nn.Linear(Lv[2] * 1 * 1, Lv[3]),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(Lv[3], out_size),
            Projector(rep_dim=out_size, proj_dim=self.proj_dim)
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
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = state_dict[key]
        self.load_state_dict(new_state_dict)
        return self

class Encoder2(nn.Module):
    def __init__(self, Lv=[64, 128, 256, 256, 256], ks=[11], out_size = 2, proj_dim=5, fc_depth=2, input_size=121):
        super(Encoder2, self).__init__()
        self.proj_dim = out_size if out_size < proj_dim else proj_dim
        self.enc_block1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=Lv[0], kernel_size=ks[0], padding=math.ceil((ks[0]-1)/2)),
            nn.BatchNorm1d(Lv[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(p=0.2),
            nn.Conv1d(Lv[0], Lv[1], ks[0], padding=math.ceil((ks[0]-1)/2)),
            nn.BatchNorm1d(Lv[1]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            # nn.Dropout(p=0.2),
            nn.Conv1d(Lv[1], Lv[2], ks[0], padding=math.ceil((ks[0]-1)/2)),
            nn.BatchNorm1d(Lv[2]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(Lv[2], Lv[3], ks[0], padding=math.ceil((ks[0]-1)/2)),
            nn.BatchNorm1d(Lv[2]),
            nn.ReLU(),
        )
        self.avgpool1d = nn.AdaptiveAvgPool1d((1))
        list_layers = [nn.Linear(Lv[3] * 1 * 1, Lv[4]), nn.ReLU(inplace=True)]
        for _ in range(fc_depth-2):
            list_layers += [nn.Linear(Lv[4], Lv[4]), nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(Lv[4], out_size), nn.ReLU(inplace=True)]
        list_layers += [Projector(rep_dim=out_size, proj_dim=self.proj_dim)]
        
        self.fcpart = nn.Sequential(*list_layers)
        
        # nn.Sequential(
        #     nn.Linear(Lv[2] * 1 * 1, Lv[3]),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.2),
        #     nn.Linear(Lv[3], out_size),
        #     )
        self.Lv = Lv
        # self.projector = Projector2(rep_dim=out_size, proj_dim=self.proj_dim)
    def forward(self, x):
        x = self.enc_block1d(x)
        # print(x.shape)
        x = self.avgpool1d(x)
        x = x.view(-1, self.Lv[2] * 1 * 1)
        x = self.fcpart(x)
        # x = self.projector(x)
        return x

class FullyConnectedEnc(nn.Module):
    def __init__(self, input_size=121, Lv=[768, 512, 256], out_size=2, proj_dim=5, fc_depth=2, multichan=False):
        super(FullyConnectedEnc, self).__init__()
        self.proj_dim = out_size if out_size < proj_dim else proj_dim
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
            Projector(rep_dim=out_size, proj_dim=self.proj_dim)
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
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = state_dict[key]
        self.load_state_dict(new_state_dict)
        return self


# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
class _Norm(nn.Module):

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp / torch.norm(inp, dim=1, keepdim=True)
    

class Squeeze(nn.Module):
    """Squeeze 3rd dimension of input tensor, pass through otherwise."""

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Squeeze 3rd dimension of input tensor, pass through otherwise.

        Args:
            inp: 1-3D input tensor

        Returns:
            If the third dimension of the input tensor can be squeezed,
            return the resulting 2D output tensor. If input is 2D or less,
            return the input.
        """
        if inp.dim() > 2:
            return inp.squeeze(2)
        return inp

    
class _Skip(nn.Module):
    """Add a skip connection to a list of modules

    Args:
        *modules (torch.nn.Module): Modules to add to the bottleneck
        crop (tuple of ints): Number of timesteps to crop around the
            shortcut of the module to match the output with the bottleneck
            layers. This can be typically inferred from the strides/sizes
            of any conv layers within the bottleneck.
    """

    def __init__(self, *modules, crop=(1, 1)):
        super().__init__()
        self.module = nn.Sequential(*modules)
        self.crop = slice(
            crop[0],
            -crop[1] if isinstance(crop[1], int) and crop[1] > 0 else None)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute forward pass through the skip connection.

        Implements the operation ``self.module(inp[..., self.crop]) + skip``.

        Args:
            inp: 3D input tensor

        Returns:
            3D output tensor of same dimension as `inp`.
        """
        skip = self.module(inp)
        return inp[..., self.crop] + skip


class CEBRA(nn.Module):
    def __init__(self, num_units=32, out_size = 2, proj_dim=5, fc_depth=2, input_size=121, multichan=False):
        super(CEBRA, self).__init__()
        print("init CEBRA Encoder")
        self.proj_dim = out_size if out_size < proj_dim else proj_dim
        self.multichan = multichan
        self.input_size = input_size
        self.enc_block1d = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=num_units, kernel_size=2),
            nn.GELU(),
            _Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            _Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            _Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(in_channels=num_units, out_channels=out_size, kernel_size=3),
            _Norm(),
            Squeeze(),
        )

        self.projector = Projector(rep_dim=out_size, proj_dim=self.proj_dim)

    def forward(self, x):
        if self.multichan:
            x = x.view(-1, 1, self.input_size)
        x = self.enc_block1d(x)
        x = self.projector(x)
        return x

    def load(self, fname_model):
        checkpoint = torch.load(fname_model, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for key in state_dict:
            # if "backbone" in key and "fc" not in key:
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = state_dict[key]
        self.load_state_dict(new_state_dict)
        return self

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, num_chans: int = 1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.num_chans = num_chans

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
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

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if self.num_chans > 1:
            x = x + self.pe[:, :, :x.size(2)]
        else:
            x = x + self.pe[:, :, :x.size(1)]

        return self.dropout(x)

class AttentionEnc(nn.Module):
    def __init__(self, spike_size=121, n_channels=1, out_size=2, proj_dim=5, fc_depth=2, nlayers=9, nhead=8, dropout=0.1, expand_dim=16, cls_head=None):
        super(AttentionEnc, self).__init__()
        self.spike_size = spike_size
        self.expand_dim = expand_dim
        self.proj_dim = out_size if out_size < proj_dim else proj_dim
        if expand_dim != 1:
            self.encoder = nn.Linear(n_channels, expand_dim)
        else:
            nhead = 1
        self.pos_encoder = PositionalEncoding(expand_dim, dropout, spike_size)
        encoder_layers = TransformerEncoderLayer(expand_dim, nhead, 512, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        list_layers = [nn.Linear(self.spike_size * expand_dim, 256), nn.ReLU(inplace=True)]
        for _ in range(fc_depth-2):
            list_layers += [nn.Linear(256, 256), nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(256, out_size)]
        if not cls_head:
            list_layers += [Projector(rep_dim=out_size, proj_dim=self.proj_dim)]
        else:
            print(f"using head = {cls_head}")
            if cls_head == 'linear':
                self.cls_head = nn.Linear(out_size, 10)
            elif cls_head == 'mlp2':
                self.cls_head = nn.Sequential(nn.Linear(out_size, 100), nn.ReLU(inplace=True), \
                    nn.Linear(100, 10))
            elif cls_head == 'mlp3':
                self.cls_head = nn.Sequential(nn.Linear(out_size, 100), nn.ReLU(inplace=True), \
                    nn.Linear(100, 50), nn.ReLU(inplace=True), \
                    nn.Linear(50, 10))         
            list_layers += [self.cls_head]
            
        self.fcpart = nn.Sequential(*list_layers)
        
        # self.fcpart = nn.Sequential(
        #     nn.Linear(self.spike_size * expand_dim, self.spike_size),
        #     nn.ReLU(),
        #     nn.Linear(self.spike_size, out_size),
            
        #     # nn.ReLU(),
        #     # nn.Dropout(p=0.2),
        #     # nn.Linear(5 * self.spike_size * expand_dim, out_size),
        #     Projector(rep_dim=out_size, proj_dim=self.proj_dim)
        # )

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
        src = torch.transpose(src, 1, 2)
        if self.expand_dim != 1:
            src = self.encoder(src) * math.sqrt(self.expand_dim)
            
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src, src_mask)
        output = output.view(-1, self.spike_size * self.expand_dim)
        output = self.fcpart(output)
        return output

    def load(self, fname_model):
        checkpoint = torch.load(fname_model, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for key in state_dict:
            # if "backbone" in key and "fc" not in key:
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = state_dict[key]
            if 'pos_encoder' in key:
                new_state_dict[new_key] = state_dict[key].transpose(0, 1)
        self.load_state_dict(new_state_dict)
        return self


class MultiChanAttentionEnc1(nn.Module):
    def __init__(self, spike_size=121, n_channels=11, out_size=2, proj_dim=5, fc_depth=2, nlayers=9, nhead=8, dropout=0.1, expand_dim=16, cls_head=None):
        super(MultiChanAttentionEnc1, self).__init__()
        self.spike_size = spike_size
        self.expand_dim = expand_dim
        self.n_channels = n_channels
        self.proj_dim = out_size if out_size < proj_dim else proj_dim
        if expand_dim != 1:
            self.encoder = nn.Linear(1, expand_dim)
        else:
            nhead = 1
        self.pos_encoder = PositionalEncoding(expand_dim, dropout, spike_size, num_chans=n_channels)
        # encoder_layers = TransformerEncoderLayer(expand_dim, nhead, 512, batch_first=True)
        encoder_layers = TransformerEncoderLayer(expand_dim, nhead, 512)
        
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder_sum = nn.Linear(n_channels, 1)
        list_layers = [nn.Linear(self.spike_size * self.expand_dim, 256), nn.ReLU(inplace=True)]
        for _ in range(fc_depth-2):
            list_layers += [nn.Linear(256, 256), nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(256, out_size)]
        if not cls_head:
            list_layers += [Projector(rep_dim=out_size, proj_dim=self.proj_dim)]
        else:
            print(f"using head = {cls_head}")
            if cls_head == 'linear':
                self.cls_head = nn.Linear(out_size, 10)
            elif cls_head == 'mlp2':
                self.cls_head = nn.Sequential(nn.Linear(out_size, 100), nn.ReLU(inplace=True), \
                    nn.Linear(100, 10))
            elif cls_head == 'mlp3':
                self.cls_head = nn.Sequential(nn.Linear(out_size, 100), nn.ReLU(inplace=True), \
                    nn.Linear(100, 50), nn.ReLU(inplace=True), \
                    nn.Linear(50, 10))
            list_layers += [self.cls_head]
            
        self.fcpart = nn.Sequential(*list_layers)
        
        # self.fcpart = nn.Sequential(
        #     nn.Linear(self.spike_size * expand_dim, self.spike_size),
        #     nn.ReLU(),
        #     nn.Linear(self.spike_size, out_size),
            
        #     # nn.ReLU(),
        #     # nn.Dropout(p=0.2),
        #     # nn.Linear(5 * self.spike_size * expand_dim, out_size),
        #     Projector(rep_dim=out_size, proj_dim=self.proj_dim)
        # )

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
        # src = torch.transpose(src, 1, 2)

        # for chan in range(self.n_channels):
        #     curr_chan = src[:, :, chan]
        #     curr_chan = torch.unsqueeze(curr_chan, dim=2)
        #     # curr_chan = torch.transpose(curr_chan, 1, 2)
        #     if self.expand_dim != 1:
        #         curr_chan = self.encoder(curr_chan) * math.sqrt(self.expand_dim)
        #     curr_chan = self.pos_encoder(curr_chan)
        #     curr_chan = self.transformer_encoder(curr_chan, src_mask)
        #     # src[:, chan] = torch.transpose(curr_chan, 1, 2)
        #     src[:, :, chan] = curr_chan
        
        if self.expand_dim != 1:
            src = torch.unsqueeze(src, dim=-1)
        src = self.pos_encoder(src)  # [B, n_chans, seq_len, embed_dim]
        src = src.reshape(-1, self.spike_size, self.expand_dim) # [B * n_chans, seq_len, embed_dim]
        src = src.permute(1, 0, 2) # remove batch_first argument needs Batch in second dim
        
        output = self.transformer_encoder(src, src_mask)
        # output = output.view(-1, self.n_channels, self.spike_size, self.expand_dim)
        # output = torch.transpose(src, 1, 2)
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
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = state_dict[key]
            if 'pos_encoder' in key:
                new_state_dict[new_key] = state_dict[key].transpose(0, 1)
        self.load_state_dict(new_state_dict)
        return self
    

model_dict = { "custom_encoder": Encoder,
                           "custom_encoder2": Encoder2,
                            "denoiser": SingleChanDenoiser,
                            "fc_encoder": FullyConnectedEnc,
                            "attention": AttentionEnc,
                            "attention_multichan": MultiChanAttentionEnc1,
                            "cebra": CEBRA,
                            # "resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            # "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)
                            }

class ModelSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, proj_dim, fc_depth=2, expand_dim=16, ckpt=None, cls_head=None, multichan=True, input_size=121):
        super(ModelSimCLR, self).__init__()
        
        base_model += '_multichan' if multichan and 'attention' in base_model else ''

        if "attention" in base_model:
            self.backbone = model_dict[base_model](out_size=out_dim, proj_dim=proj_dim, fc_depth=fc_depth, expand_dim=expand_dim, cls_head=cls_head)
        else:
            self.backbone = model_dict[base_model](out_size=out_dim, proj_dim=proj_dim, fc_depth=fc_depth, input_size=input_size, multichan=multichan)

        # self.backbone = self._get_basemodel(base_model)
        print("number of encoder params: ", sum(p.numel() for p in self.backbone.parameters()))
        print("number of transfomer params: ", sum(p.numel() for n,p in self.backbone.named_parameters() if 'transformer_encoder' in n))
        print("number of fcpart params: ", sum(p.numel() for n,p in self.backbone.named_parameters() if ('fcpart' in n and 'proj' not in n)))
        print("number of Proj params: ", sum(p.numel() for n,p in self.backbone.named_parameters() if ('fcpart' in n and 'proj' in n)))
        print("number of classifier params: ", sum(p.numel() for n,p in self.backbone.named_parameters() if 'cls_head' in n))
        
        
        if base_model == "denoiser":
            # add mlp projection head
            self.backbone.fc = nn.Sequential(self.backbone.fc, Projector(rep_dim=out_dim, proj_dim=proj_dim))

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: basic_backbone, resnet18, or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
    
