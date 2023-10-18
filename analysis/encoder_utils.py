from torch import nn
from ceed.models.model_simclr import Encoder, Encoder2, FullyConnectedEnc, AttentionEnc, MultiChanAttentionEnc1
from analysis.projections import pca_train, pca
from analysis.plotting import plot_contr_v_pca
from utils.load_models import get_backbone
from utils.utils import get_contr_representations


def get_enc_backbone(enc):
    last_layer = list(list(enc.children())[-1].children())[:-1]
    enc.fcpart = nn.Sequential(*last_layer)
    return enc


def get_ckpt_results(ckpt, lat_dim, train_data, test_data, plot=False, wfs=None, wfs_interest=None, title=None, enc_type=None, Lv=None, ks=None, fc=None, save_name=None):
    if enc_type is None or enc_type == 'encoder':
        Lv = [200, 150, 100, 75] if Lv is None else Lv
        ks = [11, 21, 31] if ks is None else ks
        enc = Encoder(Lv=Lv, ks=ks, out_size=lat_dim).load(ckpt)
        backbone = get_backbone(enc)
    elif enc_type == 'fc_encoder':
        Lv = [121, 550, 1100, 250] if Lv is None else Lv
        enc = FullyConnectedEnc(Lv=Lv, out_size=lat_dim).load(ckpt)
        backbone = get_backbone(enc)
    elif enc_type == 'custom_encoder2':
        Lv=[64, 128, 256, 256, 256] if Lv is None else Lv
        ks = [11] if ks is None else ks
        enc = Encoder2(Lv=Lv, ks=ks, out_size=lat_dim).load(ckpt)
        backbone = get_backbone(enc)
    elif enc_type == 'attention_encoder':
        fc_depth = 2 if fc is None else fc
        print(fc_depth)
        enc = AttentionEnc(out_size=lat_dim, proj_dim=5, fc_depth=fc_depth, dropout=0.1, expand_dim=16).load(ckpt)
        backbone = get_backbone(enc)
    elif enc_type == 'mc_attention_encoder':
        fc_depth = 2 if fc is None else fc
        print(fc_depth)
        enc = MultiChanAttentionEnc1(out_size=lat_dim, proj_dim=5, fc_depth=fc_depth, dropout=0.1, expand_dim=16).load(ckpt)
        backbone = get_backbone(enc)
        
    contr_reps_train = get_contr_representations(backbone, train_data)
    contr_reps_test = get_contr_representations(backbone, test_data)

    if lat_dim > 2:
        # contr_reps_test_umap = learn_manifold_umap(contr_reps_test, 2) 
        contr_reps_test_pca, _ = pca(contr_reps_test, 2)
    else:
        contr_reps_test_pca = contr_reps_test

    pca_tr, _ = pca(train_data, lat_dim)
    pca_test, _ = pca_train(train_data, test_data, lat_dim)
    
    if plot:
        plot_contr_v_pca(pca_test, contr_reps_test_pca, wfs, wfs_interest, title=title, save_name=save_name)
    
    return contr_reps_train, contr_reps_test, contr_reps_test_pca, pca_tr, pca_test
