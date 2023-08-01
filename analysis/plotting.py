from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from data_aug.wf_data_augs import AmpJitter, Jitter, Collide, Crop, SmartNoise, ElectrodeDropout
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe

import colorcet as cc
def get_ccolor(k):
    if k == -1:
        return "#808080"
    else:
        return ccolors[k % len(ccolors)]
ccolors = cc.glasbey[:31]

# %matplotlib inline
plt.rc("figure", dpi=100)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


def plot_closest_spikes(X_plot, X_dist, wfs_plot, labels, labels_plot, plot_feature='cont pc', num_spikes=5, sub_figsize=(12,6), annotate_offset = .05, close_to=1, alpha=.05):
    indices = np.where(np.in1d(labels, labels_plot))[0]
    X = X_plot

    a = X_dist[np.where(labels == labels_plot[0])]
    b = X_dist[np.where(labels == labels_plot[1])]

    pcs_a = X_plot[np.where(labels == labels_plot[0])]
    pcs_b = X_plot[np.where(labels == labels_plot[1])]
    
    distance_b = cdist(a, np.mean(b,0)[None,:], 'euclidean')
    distance_a = cdist(a, np.mean(a,0)[None,:], 'euclidean')
    
    if close_to == 1:
        min_indices = np.argsort(np.min(distance_b, axis=1))[:num_spikes]
        min_dist = np.sort(np.min(distance_b, axis=1))[:5]
    else:
        min_indices = np.argsort(np.min(distance_a, axis=1))[:num_spikes]
        min_dist = np.sort(np.min(distance_a, axis=1))[:num_spikes]

    for i, min_idx in enumerate(min_indices):
        fig = plt.figure(figsize=sub_figsize)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[4, 2])

        ax1 = plt.subplot(gs[:2, 0])
        ax1.scatter(X[indices, 0], X[indices, 1], c=[ccolors[i] for i in labels[indices]], s=40, zorder=2, alpha=alpha)
        for label_plot in np.unique(labels[indices]):
            mean_location = np.mean(X[np.where(labels == label_plot)], 0)
            ax1.scatter(mean_location[0], mean_location[1], color='black', alpha=1, s=40, zorder=2)
            ax1.annotate(f'{label_plot}', (mean_location[0] + annotate_offset, mean_location[1] + annotate_offset))
        ax1.scatter(pcs_a[min_idx, 0], pcs_a[min_idx, 1], edgecolors='black', c=ccolors[labels_plot[0]], s=40, zorder=3, alpha=1)

        ax1.set_xlabel(plot_feature + '1')
        ax1.set_ylabel(plot_feature + '2')
        dist_a = cdist(a[min_idx][None,:], np.mean(a,0)[None,:], 'euclidean')[0]
        dist_b = cdist(a[min_idx][None,:], np.mean(b,0)[None,:], 'euclidean')[0]
        if close_to == 1:
            ax1.set_title(f'dist mean({labels_plot[1]}): {np.round(min_dist[i],2)} | dist mean({labels_plot[0]}): {np.round(dist_a[0],2)}')
        else:
            ax1.set_title(f'dist mean({labels_plot[1]}): {np.round(dist_b[0],2)} | dist mean({labels_plot[0]}): {np.round(min_dist[i],2)}')

        # Second plot
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 1])

        spikes_a = wfs_plot[labels == labels_plot[0]]
        spikes_b = wfs_plot[labels == labels_plot[1]]

        ax2.plot(np.mean(spikes_a, 0).flatten(), color=ccolors[labels_plot[0]])
        ax3.plot(np.mean(spikes_b, 0).flatten(), color=ccolors[labels_plot[1]])
        ax2.plot(spikes_a[min_idx].flatten(), color='k', lw=2, path_effects=[pe.Stroke(linewidth=5, alpha=.25, foreground=ccolors[labels_plot[0]]), pe.Normal()], alpha=.25)
        ax3.plot(spikes_a[min_idx].flatten(), color='k', lw=2, path_effects=[pe.Stroke(linewidth=5, alpha=.25, foreground=ccolors[labels_plot[0]]), pe.Normal()], alpha=.25)

        ax2.set_title(f'template {labels_plot[0]}')
        ax3.set_title(f'template {labels_plot[1]}')

        plt.show()



def plot_all_pts(og_reps, title, save_name=None):
    dim = og_reps.shape[1]
    num_axes = 2 if dim > 2 else 1

    fig = plt.figure(figsize=(num_axes*5, 5))

    ax0 = fig.add_subplot(1, num_axes, 1)
    ax0.scatter(og_reps[:, 0], og_reps[:, 1], c='blue', clip_on=False)
    # ax0.scatter(tform_reps[:, 0], tform_reps[:, 1], c='blue', clip_on=False)
    ax0.set_title('first 2 dims')
    if dim > 2:
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        ax1.scatter(og_reps[:, 2], og_reps[:, 3], og_reps[:, 4], c='red', clip_on=False)
        # ax1.scatter(tform_reps[:, 2], tform_reps[:, 3], tform_reps[:, 4], c='red', clip_on=False)
        ax1.set_title('next 3 dims')
    plt.suptitle(title)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
        
        
def plot_contr_v_pca(pca_reps, contr_reps, wfs, wf_interest, title=None, save_name=None, wf_selection=None):
    og_wfs = wfs[wf_interest]
    n_temps = len(pca_reps)
    lat_dim = pca_reps.shape[1]
    num_wfs = len(og_wfs)
    
    max_chan_max = np.max(np.max(og_wfs, axis=1))
    max_chan_min = np.min(np.min(og_wfs, axis=1))
    # max_chan_max = max([np.max(temp) for temp in tot_temps])
    # max_chan_min = min([np.min(temp) for temp in tot_temps])
    if wf_selection is None:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'cyan', 'violet', 'maroon', 'pink'][:num_wfs]
    else:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'cyan', 'violet', 'maroon', 'pink'][wf_selection[0]:wf_selection[1]]
        print(colors)
    num_reps = int(len(pca_reps) / num_wfs)
    print(num_reps)
    labels = np.array([[colors[i] for j in range(num_reps)] for i in range(num_wfs)])
    labels = labels.flatten()
    print(labels.shape)
    
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = GridSpec(4, num_wfs, figure=fig)
    
    ax0 = fig.add_subplot(gs[:3, :int(num_wfs/2)])
    ax0.title.set_text('PCA wf representations')
    ax0.scatter(pca_reps[:, 0], pca_reps[:, 1], c=labels, clip_on=False)
    
    ax1 = fig.add_subplot(gs[:3, int(num_wfs/2):])
    ax1.title.set_text('Contrastive wf representations')
    ax1.scatter(contr_reps[:, 0], contr_reps[:, 1], c=labels, clip_on=True) 
    # ax1.set_xlim([0, 25])
    # ax1.set_ylim([-7, 15])
    
    axs = [fig.add_subplot(gs[3, i]) for i in range(num_wfs)]
        
    x = np.arange(0, 121)

    for i in range(num_wfs):
        # axs[0] = fig.add_subplot(gs[i//2, 2 + 2*(i%2)])
        axs[i].set_ylim(max_chan_min-0.5, max_chan_max+0.5)
        axs[i].title.set_text('unit {}'.format(str(wf_interest[i])))
        axs[i].plot(x, og_wfs[i], linewidth=2, markersize=12, color=colors[i])
        axs[i].get_xaxis().set_visible(False)
    
    # fig.subplots_adjust(wspace=0)

    fig.suptitle(title)
    
    if save_name is not None:
        plt.savefig(save_name)
        
def plot_recon_v_spike(wf_train, wf_test, wfs, wf_interest, ckpt, lat_dim, title, save_name=None, wf_selection=None):
    og_wfs = wfs[wf_interest]
    tot_spikes, n_times = wf_test.shape
    spike_sel = np.random.choice(tot_spikes)
    spike = wf_test[spike_sel]
    num_wfs = 10
    
    pca_aug = PCA_Reproj()
    pca_train = np.array([pca_aug(wf) for wf in wf_train])
    pca_test = np.array([pca_aug(wf) for wf in wf_test])
    
    _, contr_spikes_test, contr_spikes_test_pca, _, pca_spikes_test = get_ckpt_results(ckpt, lat_dim, wf_train, wf_test)
    # contr_spikes_test_pca = contr_spikes_test_pca.reshape(4, num_ex, -1)
    # pca_spikes_test = pca_spikes_test.reshape(4, num_ex, -1)
    
    _, contr_recon_test, contr_recon_test_pca, _, pca_recon_test = get_ckpt_results(ckpt, lat_dim, pca_train, pca_test)
    # contr_recon_test_pca = contr_recon_test_pca.reshape(4, num_ex, -1)
    # pca_spikes_test = pca_spikes_test.reshape(4, num_ex, -1)
    
    max_chan_max = np.max(np.max(og_wfs, axis=1))
    max_chan_min = np.min(np.min(og_wfs, axis=1))
    # max_chan_max = max([np.max(temp) for temp in tot_temps])
    # max_chan_min = min([np.min(temp) for temp in tot_temps])
    if wf_selection is None:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'cyan', 'violet', 'maroon', 'pink'][:num_wfs]
    else:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'cyan', 'violet', 'maroon', 'pink'][wf_selection[0]:wf_selection[1]]
        print(colors)
    num_reps = int(len(wf_test) / num_wfs)
    print(num_reps)
    labels = np.array([[colors[i] for j in range(num_reps)] for i in range(num_wfs)])
    labels = labels.flatten()
    print(labels.shape)
    
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = GridSpec(4, num_wfs, figure=fig)
    
    ax0 = fig.add_subplot(gs[:3, :int(num_wfs/2)])
    ax0.title.set_text('Contrastive spike representations')
    ax0.scatter(contr_spikes_test_pca[:, 0], contr_spikes_test_pca[:, 1], c=labels, clip_on=False)
    
    ax1 = fig.add_subplot(gs[:3, int(num_wfs/2):])
    ax1.title.set_text('Contrastive pca recon. spike representations')
    ax1.scatter(contr_recon_test_pca[:, 0], contr_recon_test_pca[:, 1], c=labels, clip_on=True) 
    # ax1.set_xlim([0, 25])
    # ax1.set_ylim([-7, 15])
    
    axs = [fig.add_subplot(gs[3, i]) for i in range(num_wfs)]
        
    x = np.arange(0, 121)

    for i in range(num_wfs):
        # axs[0] = fig.add_subplot(gs[i//2, 2 + 2*(i%2)])
        axs[i].set_ylim(max_chan_min-0.5, max_chan_max+0.5)
        axs[i].title.set_text('unit {}'.format(str(wf_interest[i])))
        axs[i].plot(x, og_wfs[i], linewidth=2, markersize=12, color=colors[i])
        axs[i].get_xaxis().set_visible(False)
    
    # fig.subplots_adjust(wspace=0)

    fig.suptitle(title)
    
    if save_name is not None:
        plt.savefig(save_name)
        
def plot_spike_loc_classes(locs, labels, num_classes, geom, title, save_name=None):
    true_labels = np.array([[i for j in range(300)] for i in range(num_classes)]).flatten()
    print(true_labels.shape)
    cmap = plt.cm.get_cmap('hsv', num_classes)
    colors = np.array([cmap(i) for i in labels])
    true_colors = np.array([cmap(i) for i in true_labels])
#     colors = [cmap(i) for i in range(10)]
#     colors = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'cyan', 'violet', 'maroon', 'pink']
#     alphas = np.linspace(0.1, 1, num=10)
    alphas = np.ones(num_classes)
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 20
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # fig, ax = plt.subplots(4, 6, figsize=(18, 30))
    fig = plt.figure(figsize=(10, 15), constrained_layout=True)
    fig.tight_layout()
    gs = GridSpec(10, 4, figure=fig)
    
    # assigned labels plot
    ax0 = fig.add_subplot(gs[:, :2])
    ax0.scatter(geom[:,1], geom[:,2])
    ax0.scatter(locs[:, 0], locs[:, 1], c=colors, alpha=0.1)
    ax0.set_xlabel('x')
    ax0.set_ylabel('z')
    ax0.set_title('Predicted clusters from features')
    
    # true labels plot
    ax1 = fig.add_subplot(gs[:, 2:])
    ax1.scatter(geom[:,1], geom[:,2])
    ax1.scatter(locs[:, 0], locs[:, 1], c=true_colors, alpha=0.1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')
    ax1.set_title('True clusters')
    
    fig.suptitle(title)
    fig.subplots_adjust(top=0.93)
    # fig.subplots_adjust(wspace=0.12)
    
    fig.subplots_adjust(hspace=0.2)
    
    if save_name is not None:
        plt.savefig(save_name)

    
def plot_one_mc_transform(mc_wfs, aug, n_extra_chans, save_name=None):
    n_wfs, n_chans, n_times = mc_wfs.shape
    
    mc_subsel = Crop(0.0, n_extra_chans)
    if aug == 'crop':
        aug_to_use = Crop(1.0, n_extra_chans)
        aug_title = 'Crop'
        num_rows = 3
    elif aug == 'collision':
        aug_to_use = Collide(mc_wfs=True)
        aug_title = 'Multi-Chan Collision'
        num_rows = 4
    elif aug == 'dropout':
        aug_to_use = ElectrodeDropout(prob=0.2)
        aug_title = 'Electrode Dropout'
        num_rows = 3
    elif aug == 'jitter':
        aug_to_use = Jitter()
        aug_title = 'Multi-Chan Jitter'
        num_rows = 3
    elif aug == 'sc_noise':
        aug_to_use = SmartNoise()
        aug_title = 'Multi-Chan Noise'
        num_rows = 3
    elif aug == 'sc_amp':
        aug_to_use = AmpJitter()
        aug_title = 'Multi-Chan Amp Jitter'
        num_rows = 3
    else:
        print('Augmentation not available')
        return
    
    rand_ind = np.random.choice(n_wfs)
    rand_wf = mc_wfs[rand_ind]
    print(rand_ind)
    
    sub_wf = mc_subsel(rand_wf.copy())
    if aug == 'collision':
        aug_wf, coll_wf = aug_to_use(rand_wf.copy())
        coll_wf = mc_subsel(coll_wf)
    else: 
        aug_wf = aug_to_use(rand_wf.copy())
    aug_wf = aug_wf if aug == 'shift' else mc_subsel(aug_wf)
    print(aug_wf.shape)
    
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 22
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig = plt.figure(figsize=(16, 8))
#     fig, ax = plt.subplots(9, 5, figsize=(24, 16))
    gs = GridSpec(num_rows, 5, figure=fig)
    x = np.arange(5*121)
    chan_coords = np.array([121*i for i in range(1, 5)])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.title.set_text('Flattened WF with 5 channels')
    ax0.plot(x, sub_wf.flatten(), color='blue', label='flattened spike')
    ax0.get_xaxis().set_visible(False)
    for coord in chan_coords:
        ax0.axvline(x=coord, ls='--', color='black')
    ylims = ax0.get_ylim()
    print(ylims)

    if num_rows == 3:
        ax1 = fig.add_subplot(gs[1, :])
        ax1.title.set_text('Flattened Augmented WF with 5 channels')
        ax1.plot(x, aug_wf.flatten(), color='red', label='flat augmented spike')
        ax1.get_xaxis().set_visible(False)
        for coord in chan_coords:
            ax1.axvline(x=coord, ls='--', color='black')
        ax2 = fig.add_subplot(gs[2, :])
        ax2.title.set_text('Overlaid WFs')
        ax2.plot(x, sub_wf.flatten(), color='blue')
        ax2.plot(x, aug_wf.flatten(), color='red')
        ax2.get_xaxis().set_visible(False)
        for coord in chan_coords:
            ax2.axvline(x=coord, ls='--', color='black')
    else: 
        ax1 = fig.add_subplot(gs[1, :])
        ax1.title.set_text('WF used for Collision')
        ax1.plot(x, coll_wf.flatten(), color='darkgreen')
        ax1.get_xaxis().set_visible(False)
        ax1.set_ylim(ylims)
        for coord in chan_coords:
            ax1.axvline(x=coord, ls='--', color='black')
        ax2 = fig.add_subplot(gs[2, :])
        ax2.title.set_text('Flattened Augmented WF with 5 channels')
        ax2.plot(x, aug_wf.flatten(), color='red', label='flat augmented spike')
        ax2.get_xaxis().set_visible(False)
        for coord in chan_coords:
            ax2.axvline(x=coord, ls='--', color='black')
        ax3 = fig.add_subplot(gs[3, :])
        ax3.title.set_text('Overlaid WFs')
        ax3.plot(x, sub_wf.flatten(), color='blue')
        ax3.plot(x, aug_wf.flatten(), color='red')
        ax3.get_xaxis().set_visible(False)
        for coord in chan_coords:
            ax3.axvline(x=coord, ls='--', color='black')

    fig.suptitle('Multi-Channel {} WF Augmentation'.format(aug_title))
    fig.subplots_adjust(wspace=0)
    fig.subplots_adjust(top=0.9)
    fig.subplots_adjust(hspace=0.5)
    
    if save_name is not None:
        plt.savefig(save_name)

def plot_one_sc_transform(sc_wfs, aug, save_name=None):
    n_wfs, n_times = sc_wfs.shape
    
    if aug == 'amp_jitter':
        aug_to_use = AmpJitter()
        aug_title = 'Amplitude Jitter'
        num_cols = 3
    elif aug == 'crop':
        aug_to_use = Crop(0.0, 1)
        aug_title = 'Crop'
        num_cols = 3
    elif aug == 'jitter':
        aug_to_use = Jitter()
        aug_title = 'Jitter'
        num_cols = 3
    elif aug == 'collision':
        aug_to_use = Collide()
        aug_title = 'Collision'
        num_cols = 4
    elif aug == 'noise':
        aug_to_use = SmartNoise()
        aug_title = 'Noise'
        num_cols = 3
    else:
        print('Augmentation not available')
        return
    
    rand_ind = np.random.choice(n_wfs)
    rand_wf = sc_wfs[rand_ind]
    print(rand_ind)
    
    if aug == 'collision':
        aug_wf, coll_wf = aug_to_use(rand_wf.copy())
        aug_wf = np.squeeze(aug_wf)
        coll_wf = np.squeeze(coll_wf)
    elif aug == 'crop':
        aug_wf = aug_to_use(rand_wf.copy())
        aug_wf = np.squeeze(aug_wf)
    else: 
        aug_wf = aug_to_use(rand_wf.copy())
    
    if len(aug_wf.shape) == 2:
        aug_wf = np.squeeze(aug_wf)
    
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 16
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig = plt.figure(figsize=(4*num_cols, 4))
#     fig, ax = plt.subplots(9, 5, figsize=(24, 16))
    gs = GridSpec(1, num_cols, figure=fig)
    x = np.arange(121)
#     chan_coords = np.array([121*i for i in range(1, 6)])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.title.set_text('WF')
    ax0.plot(x, rand_wf, color='blue')
    ax0.get_xaxis().set_visible(False)

    if num_cols == 3:
        ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
        ax1.title.set_text('Augmented WF')
        ax1.plot(x, aug_wf, color='red')
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        
        ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)
        ax2.title.set_text('Overlaid WFs')
        ax2.plot(x, rand_wf, color='blue')
        ax2.plot(x, aug_wf, color='red')
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
    else: 
        ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
        ax1.title.set_text('Collided WF')
        ax1.plot(x, coll_wf, color='darkgreen')
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        
        ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)
        ax2.title.set_text('Augmented WF')
        ax2.plot(x, aug_wf, color='red')
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        
        ax3 = fig.add_subplot(gs[0, 3], sharey=ax0)
        ax3.title.set_text('Overlaid WFs')
        ax3.plot(x, rand_wf, color='blue')
        ax3.plot(x, aug_wf, color='red')
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)

    fig.suptitle('Single-Channel {} WF Augmentation'.format(aug_title))
    fig.subplots_adjust(wspace=0)
    fig.subplots_adjust(top=0.85)
    fig.subplots_adjust(hspace=0.5)
    
    if save_name is not None:
        plt.savefig(save_name)
        
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='gist_rainbow', zorder=2, alpha=.2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2, alpha=.2)
    # ax.axis('equal')
    
    w_factor = .2
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos[:2], covar[:2,:2], alpha=w_factor, ax=ax)
