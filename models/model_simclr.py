import math
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from exceptions.exceptions import InvalidBackboneError
from collections import OrderedDict
import sys

class SingleChanDenoiser(nn.Module):
    """Cleaned up a little. Why is conv3 here and commented out in forward?"""

    def __init__(
        self, n_filters=[16, 8, 4], filter_sizes=[5, 11, 21], spike_size=121, out_size=2
    ):
        super(SingleChanDenoiser, self).__init__()
        feat1, feat2, feat3 = n_filters
        size1, size2, size3 = filter_sizes
        self.conv1 = nn.Sequential(nn.Conv1d(1, feat1, size1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(feat1, feat2, size2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(feat2, feat3, size3), nn.ReLU())
        n_input_feat = feat2 * (spike_size - size1 - size2 + 2)
        self.fc = nn.Linear(n_input_feat, out_size)

    def forward(self, x):
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

    def load(self, fname_model):
        checkpoint = torch.load(fname_model, map_location="cpu")
        self.load_state_dict(checkpoint)
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
    
def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes)


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
    

model_dict = { "custom_encoder": Encoder,
                           "custom_encoder2": Encoder2,
                            "denoiser": SingleChanDenoiser,
                            "fc_encoder": FullyConnectedEnc,
                            "attention": AttentionEnc,
                            "attention_multichan": MultiChanAttentionEnc1,
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
    
