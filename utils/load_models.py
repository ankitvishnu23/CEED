import torch
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, WFDataset_lab
from models.model_GPT import GPTConfig, Multi_GPT, Single_GPT, Projector
from data_aug.wf_data_augs import Crop
import os

class Encoder(torch.nn.Module):
    def __init__(self, multi_chan = False, rep_dim=5, proj_dim=5, pos_enc='conseq', rep_after_proj=False, use_chan_pos=False, use_merge_layer=False, add_layernorm=False, half_embed_each=False, concat_pos=False, num_classes=10):
        super().__init__()
        if multi_chan:
            if concat_pos:
                model_args = dict(bias=False, block_size=1342, n_layer=20, n_head =4, n_embd=64, dropout=0.2, out_dim=rep_dim, proj_dim=proj_dim, is_causal=True, pos = pos_enc, multi_chan=True, use_chan_pos=use_chan_pos, use_merge_layer=use_merge_layer, add_layernorm=add_layernorm, half_embed_each=half_embed_each, concat_pos=concat_pos, num_classes=num_classes)
            else:    
                model_args = dict(bias=False, block_size=1331, n_layer=20, n_head =4, n_embd=64, dropout=0.2, out_dim=rep_dim, proj_dim=proj_dim, is_causal=True, pos = pos_enc, multi_chan=True, use_chan_pos=use_chan_pos, use_merge_layer=use_merge_layer, add_layernorm=add_layernorm, half_embed_each=half_embed_each, concat_pos=concat_pos, num_classes=num_classes)
        else:
            model_args = dict(bias=False, block_size=121, n_layer=20, n_head =4, n_embd=32, dropout=0.0, out_dim=rep_dim, proj_dim=proj_dim, is_causal=True, pos = pos_enc, multi_chan=False, num_classes=num_classes)
        gptconf = GPTConfig(**model_args)
        if multi_chan:
            self.backbone = Multi_GPT(gptconf)
        else:
            self.backbone = Single_GPT(gptconf)
            self.backbone.projector = torch.nn.Identity() # this is loaded separately later
        if rep_after_proj:
            self.projector = Projector(rep_dim=gptconf.out_dim, proj_dim=gptconf.proj_dim)
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

def load_ckpt(ckpt_path, multi_chan=False, rep_dim=5, proj_dim=5, pos_enc='conseq', rep_after_proj=False, use_chan_pos=False, use_merge_layer=False, add_layernorm=False, half_embed_each=False, concat_pos=False, num_classes=10):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model = Encoder(multi_chan=multi_chan, rep_dim=rep_dim, proj_dim=proj_dim, pos_enc=pos_enc, rep_after_proj=rep_after_proj, use_chan_pos=use_chan_pos, use_merge_layer=use_merge_layer, add_layernorm=add_layernorm, half_embed_each=half_embed_each, concat_pos=concat_pos, num_classes=num_classes)
    if multi_chan:
        state_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        m, uek = model.load_state_dict(state_dict, strict=False)
#         if rep_after_proj:
#             for k in uek:
#                 assert 'online_head' in k, "Error: key matching errors!"
#         else:
#             for k in uek:
#                 assert 'projector' in k or 'online' in k, "Error: key matching errors!"
    else:
        state_dict = {'backbone.'+k: v for k,v in ckpt['state_dict'].items() if 'projector' not in k}
        state_dict.update({k: v for k,v in ckpt['state_dict'].items() if 'projector' in k})
        m, uek = model.load_state_dict(state_dict, strict=False)
#         if not rep_after_proj:
#             for k in uek:
#                 assert 'projector' in k, "Error: key matching errors!"
#         else:
#             assert(len(uek)==0)
    # assert(len(m)==0)
    print("missing keys", m)
    print("unexpected keys", uek)
    # assert that unexpected keys should only contain the string 'projector'
    
    return model

def get_dataloader(data_path, multi_chan=False, split='train', use_chan_pos=False):
    if multi_chan:
        dataset = WFDataset_lab(data_path, split=split, multi_chan=True, use_chan_pos=use_chan_pos,transform=Crop(prob=0.0, num_extra_chans=5, ignore_chan_num=True))
    else:
        print("Loading single channel data")
        dataset = WFDataset_lab(data_path, split=split, multi_chan=False)
        
    loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=False,
            num_workers=16, pin_memory=True, drop_last=False)
    return loader


def save_reps(model, loader, ckpt_path, split='train', multi_chan=False,rep_after_proj=False, use_chan_pos=False, suffix=''):
    ckpt_root_dir = '/'.join(ckpt_path.split('/')[:-1])
    model.eval()
    feature_bank = []
    with torch.no_grad():
        for data, target in loader:
            if not multi_chan:
                data = torch.squeeze(data, dim=1)
                data = torch.unsqueeze(data, dim=-1)
            else:
                if use_chan_pos:
                    data,chan_pos = data

                data = data.view(-1, 11*121)
                data = torch.unsqueeze(data, dim=-1)
            
            if use_chan_pos:
                feature = model(data.cuda(non_blocking=True), chan_pos=chan_pos.cuda(non_blocking=True))
            else:
                feature = model(data.cuda(non_blocking=True))
            feature_bank.append(feature)
            
        feature_bank = torch.cat(feature_bank, dim=0)
        print(feature_bank.shape)
        if rep_after_proj:
            torch.save(feature_bank, os.path.join(ckpt_root_dir, f'{split}_aftproj_reps{suffix}.pt'))
            print(f"saved {split} features to {ckpt_root_dir}/{split}_aftproj_reps{suffix}.pt")
        else:
            torch.save(feature_bank, os.path.join(ckpt_root_dir, f'{split}_reps{suffix}.pt'))
            print(f"saved {split} features to {ckpt_root_dir}/{split}_reps{suffix}.pt")
