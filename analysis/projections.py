import umap.umap_ as umap
from sklearn.decomposition import PCA
from analysis.denoising import SingleChanDenoiser
from tqdm import tqdm
import numpy as np
import torch

def learn_manifold_umap(data, umap_dim, umap_min_dist=0.2, umap_metric='euclidean', umap_neighbors=10):
    md = float(umap_min_dist)
    return umap.UMAP(random_state=0, metric=umap_metric, n_components=umap_dim, n_neighbors=umap_neighbors,
                    min_dist=md).fit_transform(data)

def pca_train(train, test, n_comps):
    pca_ = PCA(n_components=n_comps, whiten=True)
    pca_.fit(train)
    test_comps = pca_.transform(test)
    return test_comps, pca_.explained_variance_ratio_

def pca(S, n_comps):
    pca_ = PCA(n_components=n_comps, whiten=True)
    return pca_.fit_transform(S), pca_.explained_variance_ratio_, pca_

def get_flattened_data(dataset, denoise=False, denoise_path=''):
    if denoise:
        denoiser = SingleChanDenoiser().load(denoise_path)
    wfs_full = []
    labels_full = []
    for i in tqdm(range(len(dataset))):
        curr_wf, curr_label = dataset[i]
        if denoise:
            with torch.no_grad():
                for j in range(curr_wf.shape[0]):
                    curr_wf[j] = denoiser(torch.from_numpy(curr_wf[j].reshape(1, 121))).cpu().numpy()
        curr_wf = curr_wf.flatten().squeeze().astype('float32')
        wfs_full.append(curr_wf)
        labels_full.append(curr_label)
    wfs_full = np.array(wfs_full)
    labels_full = np.array(labels_full)

    return wfs_full, labels_full

def compute_reps_test(model, test_wfs):
    og_reps = []
    model = model.double()
    for i, og_temp in enumerate(test_wfs):
        with torch.no_grad():
            og_rep = model(torch.from_numpy(og_temp.reshape(1, -1)).double())
        og_reps.append(og_rep.numpy())
    
    return np.squeeze(np.array(og_reps))

# og_pca, og_pca_var = pca(max_chan_hptp_temps, 2)
# tform_pca, tform_var = pca(tform_temps_numpy, 2)
# og_reps_pca, og_reps_var = pca(og_reps, 2)
# tform_reps_pca, tform_reps_var = pca(tform_reps, 2)

# og_reps_umap = learn_manifold_umap(og_reps, 2)
# tform_reps_umap = learn_manifold_umap(tform_reps, 2)
