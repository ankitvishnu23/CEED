import umap.umap_ as umap
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np

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

def get_flattened_data(dataset):
    wfs_full = []
    labels_full = []
    for i in tqdm(range(len(dataset))):
        curr_wf, curr_label = dataset[i].astype('float32')
        curr_wf = curr_wf.flatten().squeeze()
        wfs_full.append(curr_wf)
        labels_full.append(curr_label)
    wfs_full = np.array(wfs_full)
    labels_full = np.array(labels_full)

    return wfs_full, labels_full

# og_pca, og_pca_var = pca(max_chan_hptp_temps, 2)
# tform_pca, tform_var = pca(tform_temps_numpy, 2)
# og_reps_pca, og_reps_var = pca(og_reps, 2)
# tform_reps_pca, tform_reps_var = pca(tform_reps, 2)

# og_reps_umap = learn_manifold_umap(og_reps, 2)
# tform_reps_umap = learn_manifold_umap(tform_reps, 2)
