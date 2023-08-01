import spikeinterface.full as si
import MEArec as mr
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
import math
import os
import h5py
import random
import logging
import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

try:
    from brainbox.io.one import SpikeSortingLoader
    from ibllib.atlas import AllenAtlas
    from one.api import ONE
    import brainbox.io.one as bbone
except ImportError:
    print("Failed to import IBL packages (brainbox, ibllib, one, brainbox")
    
import datetime
from spike_psvae.subtract import read_geom_from_meta
from pathlib import Path
from spike_psvae.waveform_utils import make_channel_index, make_contiguous_channel_index
from spike_psvae import denoise, snr_templates, spike_train_utils
import torch
from spike_psvae.spikeio import read_waveforms

def kill_signal(recordings, threshold, window_size):
    """
    Thresholds recordings, values above 'threshold' are considered signal
    (set to 0), a window of size 'window_size' is drawn around the signal
    points and those observations are also killed
    Returns
    -------
    recordings: numpy.ndarray
        The modified recordings with values above the threshold set to 0
    is_noise_idx: numpy.ndarray
        A boolean array with the same shap as 'recordings' indicating if the
        observation is noise (1) or was killed (0).
    """
    recordings = np.copy(recordings)

    T, C = recordings.shape
    R = int((window_size-1)/2)

    # this will hold a flag 1 (noise), 0 (signal) for every obseration in the
    # recordings
    is_noise_idx = np.zeros((T, C))

    # go through every neighboring channel
    for c in range(C):

        # get obserations where observation is above threshold
        idx_temp = np.where(np.abs(recordings[:, c]) > threshold)[0]

        if len(idx_temp) == 0:
            is_noise_idx[:, c] = 1
            continue

        # shift every index found
        for j in range(-R, R+1):

            # shift
            idx_temp2 = idx_temp + j

            # remove indexes outside range [0, T]
            idx_temp2 = idx_temp2[np.logical_and(idx_temp2 >= 0,
                                                 idx_temp2 < T)]

            # set surviving indexes to nan
            recordings[idx_temp2, c] = np.nan

        # noise indexes are the ones that are not nan
        # FIXME: compare to np.nan instead
        is_noise_idx_temp = (recordings[:, c] == recordings[:, c])

        # standarize data, ignoring nans
        recordings[:, c] = recordings[:, c]/np.nanstd(recordings[:, c])

        # set non noise indexes to 0 in the recordings
        recordings[~is_noise_idx_temp, c] = 0

        # save noise indexes
        is_noise_idx[is_noise_idx_temp, c] = 1

    return recordings, is_noise_idx


def noise_whitener(recordings, temporal_size, window_size, sample_size=1000,
                   threshold=3.0, max_trials_per_sample=10000,
                   allow_smaller_sample_size=False):
    """Compute noise temporal and spatial covariance
    Parameters
    ----------
    recordings: numpy.ndarray
        Recordings
    temporal_size:
        Waveform size
    sample_size: int
        Number of noise snippets of temporal_size to search
    threshold: float
        Observations below this number are considered noise
    Returns
    -------
    spatial_SIG: numpy.ndarray
    temporal_SIG: numpy.ndarray
    """

    # kill signal above threshold in recordings
    print('Get Noise Floor')
    rec, is_noise_idx = kill_signal(recordings, threshold, window_size)

    # compute spatial covariance, output: (n_channels, n_channels)
    print('Compute Spatial Covariance')
    spatial_cov = np.divide(np.matmul(rec.T, rec),
                            np.matmul(is_noise_idx.T, is_noise_idx))
    spatial_cov[np.isnan(spatial_cov)] = 0
    spatial_cov[np.isinf(spatial_cov)] = 0

    # compute spatial sig
    w_spatial, v_spatial = np.linalg.eig(spatial_cov)
    spatial_SIG = np.matmul(np.matmul(v_spatial,
                                      np.diag(np.sqrt(w_spatial))),
                            v_spatial.T)

    # apply spatial whitening to recordings
    print('Compute Temporal Covaraince')
    spatial_whitener = np.matmul(np.matmul(v_spatial,
                                           np.diag(1/np.sqrt(w_spatial))),
                                 v_spatial.T)
    #print ("rec: ", rec, ", spatial_whitener: ", spatial_whitener.shape)
    rec = np.matmul(rec, spatial_whitener)

    # search single noise channel snippets
    noise_wf = search_noise_snippets(
        rec, is_noise_idx, sample_size,
        temporal_size,
        channel_choices=None,
        max_trials_per_sample=max_trials_per_sample,
        allow_smaller_sample_size=allow_smaller_sample_size)

    w, v = np.linalg.eig(np.cov(noise_wf.T))

    temporal_SIG = np.matmul(np.matmul(v, np.diag(np.sqrt(w))), v.T)

    return spatial_SIG, temporal_SIG


def search_noise_snippets(recordings, is_noise_idx, sample_size,
                          temporal_size, channel_choices=None,
                          max_trials_per_sample=100000,
                          allow_smaller_sample_size=False):
    """
    Randomly search noise snippets of 'temporal_size'
    Parameters
    ----------
    channel_choices: list
        List of sets of channels to select at random on each trial
    max_trials_per_sample: int, optional
        Maximum random trials per sample
    allow_smaller_sample_size: bool, optional
        If 'max_trials_per_sample' is reached and this is True, the noise
        snippets found up to that time are returned
    Raises
    ------
    ValueError
        if after 'max_trials_per_sample' trials, no noise snippet has been
        found this exception is raised
    Notes
    -----
    Channels selected at random using the random module from the standard
    library (not using np.random)
    """

    T, C = recordings.shape

    if channel_choices is None:
        noise_wf = np.zeros((sample_size, temporal_size))
    else:
        lenghts = set([len(ch) for ch in channel_choices])

        if len(lenghts) > 1:
            raise ValueError('All elements in channel_choices must have '
                             'the same length, got {}'.format(lenghts))

        n_channels = len(channel_choices[0])
        noise_wf = np.zeros((sample_size, temporal_size, n_channels))

    count = 0

    print('Starting to search noise snippets...')

    trial = 0

    # repeat until you get sample_size noise snippets
    while count < sample_size:

        # random number for the start of the noise snippet
        t_start = np.random.randint(T-temporal_size)

        if channel_choices is None:
            # random channel
            ch = random.randint(0, C - 1)
        else:
            ch = random.choice(channel_choices)

        t_slice = slice(t_start, t_start+temporal_size)

        # get a snippet from the recordings and the noise flags for the same
        # location
        snippet = recordings[t_slice, ch]
        snipped_idx_noise = is_noise_idx[t_slice, ch]

        # check if all observations in snippet are noise
        if snipped_idx_noise.all():
            # add the snippet and increase count
            noise_wf[count] = snippet
            count += 1
            trial = 0

            print('Found %i/%i...', count, sample_size)

        trial += 1

        if trial == max_trials_per_sample:
            if allow_smaller_sample_size:
                return noise_wf[:count]
            else:
                raise ValueError("Couldn't find snippet {} of size {} after "
                                 "{} iterations (only {} found)"
                                 .format(count + 1, temporal_size,
                                         max_trials_per_sample,
                                         count))

    return noise_wf

def split_data(data, num_train, num_val, num_test, n_chans, last_dim):
    train_set = []
    val_set = []
    test_set = []
    tot_num = num_train + num_val + num_test
    n_div = int(len(data) / tot_num)
    for i in range(n_div):
        start = (num_train+num_val+num_test)*i
        train_set.append(data[start:start+num_train])
        val_set.append(data[start+num_train:start+num_train+num_val])
        test_set.append(data[start+num_train+num_val:start+num_train+num_val+num_test])

    if last_dim > 0:
        train_set = np.array(train_set).reshape(-1, n_chans, last_dim)
        val_set = np.array(val_set).reshape(-1, n_chans, last_dim)
        test_set = np.array(test_set).reshape(-1, n_chans, last_dim)
    else:
        train_set = np.array(train_set).reshape(-1, n_chans)
        val_set = np.array(val_set).reshape(-1, n_chans)
        test_set = np.array(test_set).reshape(-1, n_chans)
        
    return train_set, val_set, test_set

def pad_channels(wf, geoms, mc_start, mc_end, n_chans, spike_length_samples=121):
    curr_n_chans = mc_end - mc_start
    pad_beg = n_chans - curr_n_chans if mc_start == 0 else 0
    pad_end = n_chans - curr_n_chans if mc_start > 0 else 0
    
    wf_len = wf.shape[0]
    # if pad_beg + wf_len + pad_end != n_chans:
    #     print(mc_start)
    #     print(mc_end)
    #     print(pad_beg)
    #     print(pad_end)
    #     raise ValueError("bad wf")
    
    pad_beg_wf = np.zeros((pad_beg, spike_length_samples))
    pad_end_wf = np.zeros((pad_end, spike_length_samples))
    wf = np.concatenate([pad_beg_wf, wf, pad_end_wf])
    
    pad_beg_cn = -1 * np.ones((pad_beg,))
    pad_end_cn = -1 * np.ones((pad_end,))
    chan_nums = np.concatenate([pad_beg_cn, np.arange(mc_start, mc_end), pad_end_cn])
    
    pad_beg_geom = np.zeros((pad_beg,2))
    pad_end_geom = np.zeros((pad_end,2))
    geoms = np.concatenate([pad_beg_geom, geoms, pad_end_geom])
    
    pad_beg_cn = np.zeros((pad_beg,))
    pad_end_cn = np.zeros((pad_end,))
    mask = np.concatenate([pad_beg_cn, np.ones((curr_n_chans)), pad_end_cn])
    
    return wf, chan_nums, geoms, mask

def save_sim_covs(rec_path, save_path, spike_length_samples=121):
    recgen = mr.load_recordings(rec_path, load_waveforms=False)
    
    rec = si.MEArecRecordingExtractor(rec_path)
    rec = si.bandpass_filter(rec, dtype='float32')
    rec = si.common_reference(rec, reference='global', operator='median')
    rec = si.zscore(rec)
    
    norm_chan_recording = rec.get_traces()
    
    spatial_cov, temporal_cov = noise_whitener(norm_chan_recording, spike_length_samples, 50)
    
    np.save(os.path.join(save_path, 'spatial_cov_example.npy'), spatial_cov)
    np.save(os.path.join(save_path, 'temporal_cov_example.npy'), temporal_cov)
    
    
# def save_real_covs(rec_path, save_path):
#     recgen = mr.load_recordings(rec_path, load_waveforms=False)
    
#     rec = si.MEArecRecordingExtractor(rec_path)
#     rec = si.bandpass_filter(rec, dtype='float32')
#     rec = si.common_reference(rec, reference='global', operator='median')
#     rec = si.zscore(rec)
    
#     norm_chan_recording = rec.get_traces()
    
#     spatial_cov, temporal_cov = noise_whitener(norm_chan_recording, spike_length_samples, 50)
    
#     np.save(os.path.join(save_path, '/spatial_cov.npy'), spatial_cov)
#     np.save(os.path.join(save_path, '/temporal_cov.npy'), temporal_cov)
    

def extract_IBL(bin_fp, meta_fp, pid, t_window, use_labels=True, sampling_frequency=30_000):
    one = ONE()
    ba = AllenAtlas()

    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    
    geom = read_geom_from_meta(Path(meta_fp))
    spike_times = spikes['times']
    spike_frames = sl.samples2times(spike_times, direction='reverse').astype('int')
    spike_train = np.concatenate((spike_frames.copy()[:,None], spikes['clusters'].copy()[:,None]), axis=1)
    in_rec_idxs = np.where((spike_frames >= t_window[0]*sampling_frequency) & (spike_frames <= t_window[1]*sampling_frequency))[0]
    spike_train = spike_train[in_rec_idxs, :]
    spike_train[:, 0] = spike_train[:, 0] - t_window[0]*sampling_frequency

    channel_index = make_contiguous_channel_index(geom.shape[0], n_neighbors=40)
    closest_channel_list = []
    for cluster_id in spikes['clusters']:
        closest_channel = clusters['channels'][cluster_id]
        closest_channel_list.append(closest_channel)
    closest_channels = np.asarray(closest_channel_list)
    closest_channels = closest_channels[in_rec_idxs]
    
    aligned_spike_train, order, templates, template_shifts = spike_train_utils.clean_align_and_get_templates(spike_train, geom.shape[0], bin_fp)
    templates, _ = snr_templates.get_templates(aligned_spike_train, geom, bin_fp, closest_channels, reducer=np.median)
    mcs = np.array([templates[unit_id].ptp(0).argmax(0) for unit_id in range(len(templates))])
    
    spike_index = aligned_spike_train
    if use_labels:
        spike_index = np.vstack([spike_index[:, 0], np.array([mcs[unit] for unit in aligned_spike_train[:, 1]]), spike_index[:, 1]])
    else:
        spike_index[:, 1] = np.array([mcs[unit] for unit in aligned_spike_train[:, 1]])
    
    return spike_index, geom, channel_index, templates
    
    
def extract_sim(rec_path, wfs_per_unit, use_labels=True, geom_dims=(1,2), trough_offset=42, spike_length_samples=131, random_seed=0):
    recgen = mr.load_recordings(rec_path, load_waveforms=False)
    # recgen.extract_templates(cut_out=[1.9,1.91], recompute=True)
    geom_original = recgen.channel_positions[()]
    depth_order = np.argsort(geom_original[:,geom_dims[1]])
    geom = geom_original[depth_order]
    sort = si.MEArecSortingExtractor(rec_path)
    
    rec = si.MEArecRecordingExtractor(rec_path)
    rec = si.bandpass_filter(rec, dtype='float32')
    rec = si.common_reference(rec, reference='global', operator='median')
    rec = si.zscore(rec)
    
    pre_peak = trough_offset+5
    post_peak = spike_length_samples - pre_peak

    folder = 'waveform_folder'
    we = si.extract_waveforms(
        rec,
        sort,
        folder,
        ms_before=(1/rec.sampling_frequency)*pre_peak*1000,
        ms_after=(1/rec.sampling_frequency)*post_peak*1000,
        max_spikes_per_unit=wfs_per_unit,
        overwrite=True,
        seed=random_seed
    )
    templates = we.get_all_templates()
    templates = templates[:, :, depth_order]
    mcs = np.array([np.argmax(template.ptp(0)) for template in templates])
    spike_train = sort.get_all_spike_trains()[0][0]
    spike_units = np.array([int(unit[1:]) for unit in sort.get_all_spike_trains()[0][1]])
    spike_unit_mcs = np.array([mcs[unit] for unit in spike_units])
    spike_index = np.vstack([spike_train, spike_units, spike_unit_mcs]) if use_labels else np.vstack([spike_train, spike_unit_mcs])
    
    return spike_index.T, geom, we


def chunk_data(spike_index, max_proc_len=25000):
    spike_idx_len = len(spike_index)
    n_chunks = spike_idx_len // max_proc_len + 1
    chunks = []
    for i in range(n_chunks):
        start = i * max_proc_len
        chunk = spike_index[start:start+max_proc_len]
        chunks.append(chunk)
    
    return chunks


def make_dataset(bin_path, spike_index, geom, save_path, geom_dims=(1,2), we=None, templates=None,
                 num_chans_extract=21, channel_index=None, unit_ids=None, train_num=1200, val_num=100,
                 test_num=200, trough_offset=42, spike_length_samples=121, do_split=True, plot=False, random_seed=0):
    np.random.seed(random_seed)
    num_waveforms = train_num + val_num + test_num
    spikes_array = []
    geom_locs_array = []
    max_chan_array = []
    max_proc_len = 25000
    num_template_amps_shift = 4
    num_chans = math.floor(num_chans_extract/2)
    tot_num_chans = geom.shape[0]
    if we is not None:
        depth_order = np.argsort(geom[:,geom_dims[1]])
        geom = geom[depth_order]
    num_waveforms = train_num + val_num + test_num

    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 22
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    if plot:
        if unit_ids is not None:
            fig = plt.figure(figsize=(12 + int(num_chans_extract*0.5), 24))
            gs = GridSpec(len(unit_ids), num_chans_extract+1, figure=fig)
            x = np.arange(spike_length_samples)
        else:
            print("no unit ids, skipping plotting...")
    
    if unit_ids is None:
        print("chunking data")
        data_chunks = chunk_data(spike_index, max_proc_len)
        for i in range(len(data_chunks)):
            curr_chunk = data_chunks[i]
            if i == 0:
                print("reading waveforms")
            waveforms, _ = read_waveforms(curr_chunk, bin_path, 
                                      n_channels=geom.shape[0], spike_length_samples=121)
            mcs = curr_chunk[:, 1]
            if i == 0:
                print("cropping and storing waveforms")
            for i, waveform in enumerate(waveforms):
                mc_curr = mcs[i]
                mc_start = max(mc_curr - num_chans, 0)
                mc_end = min(mc_curr + num_chans, tot_num_chans) + 1
                crop_wf, crop_chan, crop_geom, mask = pad_channels(waveform.T[mc_start:mc_end], 
                                                                   geom[mc_start:mc_end, 1:], mc_start, mc_end, 
                                                                   num_chans_extract)
                spikes_array.append(crop_wf)
                geom_locs_array.append(crop_geom)
                max_chan_array.append(crop_chan)
        # max_chan_array = mcs
    else:
        for k, unit_id in tqdm(enumerate(unit_ids), desc="Unit IDs"):
            curr_temp_wfs = []
            curr_geom_locs = []
            curr_spike_max_chan = []
            
            if len(spike_index[:,0][np.where(spike_index[:,1]==unit_id)[0]]) < num_waveforms:
                continue
            if we is not None:
                waveforms = we.get_waveforms("#{}".format(str(unit_id)))
                waveforms = waveforms[:num_waveforms, :, depth_order]
                templates = we.get_all_templates()
                templates = templates[:, :, depth_order]
            else:
                spike_frames_template = np.random.choice(spike_index[:,0][np.where(spike_index[:,2] == i)[0]], 
                                                         size=num_waveforms)
                waveforms, _ = read_waveforms(spike_frames_template, bin_path, channel_index=channel_index,
                                              n_channels=geom.shape[0], spike_length_samples=spike_length_samples)
            mc = np.unique(spike_index[:, 2][np.where(spike_index[:,1] == unit_id)[0]])[0]
            mcs_template = templates[unit_id].ptp(0).argsort()[::-1][:num_template_amps_shift]
            shifts = np.abs(waveforms[:,:,mcs_template]).max(1).argmax(1)
            mcs_shifted = mcs_template[shifts]
            for i, waveform in enumerate(waveforms):
                mc_curr = mcs_shifted[i]
                mc_start = max(mc_curr - num_chans, 0)
                mc_end = min(mc_curr + num_chans + 1, tot_num_chans)
                crop_wf, crop_chan, crop_geom, mask = pad_channels(waveform.T[mc_start:mc_end],
                                                                   geom[mc_start:mc_end, geom_dims], mc_start, 
                                                                   mc_end, num_chans_extract)
                curr_temp_wfs.append(crop_wf)
                curr_geom_locs.append(crop_geom)
                curr_spike_max_chan.append(crop_chan)

            curr_temp_wfs = np.asarray(curr_temp_wfs)
            curr_geom_locs = np.asarray(curr_geom_locs)

            spikes_array.append(curr_temp_wfs)
            geom_locs_array.append(curr_geom_locs)
            max_chan_array.append(curr_spike_max_chan)

            if plot:
                ax0 = fig.add_subplot(gs[k, 0])
                ax0.title.set_text('Unit {}'.format(str(unit_id)))
                ax0.plot(x, templates[unit_id, :, mc])
                ax0.axes.get_xaxis().set_visible(False)
                ax1 = fig.add_subplot(gs[k, 1:], sharey=ax0)
                for waveform in curr_temp_wfs[:100]:
                    ax1.plot(waveform.flatten(), color='blue', alpha=.1)
                    ax1.plot(templates[unit_id].T[mc_start:mc_end].flatten(), color='red')
                    ax1.axes.get_yaxis().set_visible(False)
                    ax1.axes.get_xaxis().set_visible(False)
        
        labels_train = np.array([[i for j in range(train_num)] for i in range(len(unit_ids))]).flatten()
        labels_val = np.array([[i for j in range(val_num)] for i in range(len(unit_ids))]).flatten()
        labels_test = np.array([[i for j in range(test_num)] for i in range(len(unit_ids))]).flatten()

        np.save(os.path.join(save_path, 'labels_train.npy'), labels_train)
        np.save(os.path.join(save_path, 'labels_val.npy'), labels_val)
        np.save(os.path.join(save_path, 'labels_test.npy'), labels_test)
        fig.subplots_adjust(wspace=0, hspace=0.25)
    
    spikes_array = np.array(spikes_array)
    geom_locs_array = np.array(geom_locs_array)
    max_chan_array = np.array(max_chan_array)
    np.save(os.path.join(save_path, 'full_raw_spikes.npy'), spikes_array)
    np.save(os.path.join(save_path, 'channel_spike_locs.npy'), geom_locs_array)
    
    print("making train, val, test splits")
    if do_split:
        train_set, val_set, test_set = split_data(spikes_array, train_num, val_num, test_num, 
                                                  num_chans_extract, last_dim=spike_length_samples)
        train_geom_locs, val_geom_locs, test_geom_locs = split_data(geom_locs_array, train_num, val_num, 
                                                                    test_num, num_chans_extract,
                                                                    last_dim=geom.shape[1])
        train_max_chan, val_max_chan, test_max_chan = split_data(max_chan_array, train_num, val_num, 
                                                                 test_num, num_chans_extract, last_dim=0)
        print("saving split results")
        np.save(os.path.join(save_path, 'spikes_train.npy'), train_set)
        np.save(os.path.join(save_path, 'spikes_val.npy'), val_set)
        np.save(os.path.join(save_path, 'spikes_test.npy'), test_set)

        np.save(os.path.join(save_path, 'channel_spike_locs_train.npy'), train_geom_locs)
        np.save(os.path.join(save_path, 'channel_spike_locs_val.npy'), val_geom_locs)
        np.save(os.path.join(save_path, 'channel_spike_locs_test.npy'), test_geom_locs)

        np.save(os.path.join(save_path, 'channel_num_train.npy'), train_max_chan)
        np.save(os.path.join(save_path, 'channel_num_val.npy'), val_max_chan)
        np.save(os.path.join(save_path, 'channel_num_test.npy'), test_max_chan)
        
        np.save(os.path.join(save_path, 'geom.npy'), geom)
        
        return train_set, val_set, test_set, train_geom_locs, val_geom_locs, test_geom_locs, train_max_chan, val_max_chan, test_max_chan
        
    else:
        #save out full dataset with no splits (useful for inference)
        print("saving no split results")
        np.save(os.path.join(save_path, 'spikes_test.npy'), spikes_array)
        np.save(os.path.join(save_path, 'channel_spike_locs_test.npy'), geom_locs_array)
        np.save(os.path.join(save_path, 'channel_num_test.npy'), max_chan_array)
    
        np.save(os.path.join(save_path, 'geom.npy'), geom)
        
        return spikes_array, geom_locs_array, max_chan_array
    
    