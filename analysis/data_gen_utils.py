import spikeinterface.full as si
import MEArec as mr
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
import numpy as np
import shutil
from scipy.spatial.distance import pdist, squareform
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm.auto import trange
import shutil

try:
    import brainbox.io.one as bbone
    from brainbox.io.one import SpikeSortingLoader
    from brainbox.io.spikeglx import Streamer
    from ibllib.atlas import AllenAtlas
    from neurodsp import voltage
    from neurodsp.utils import rms
    import spikeglx
    from one.api import ONE
except ImportError:
    print("Failed to import IBL packages (brainbox, ibllib, one, brainbox")

try:
    from spike_psvae.subtract import read_geom_from_meta
    from spike_psvae.waveform_utils import make_contiguous_channel_index
    from spike_psvae import snr_templates, spike_train_utils
    from spike_psvae.spikeio import read_waveforms
except ImportError:
    print("Failed to import spike-psvae/dartsort functions")

def kill_signal(recordings, threshold, window_size):
    """
    Thresholds recordings, values above 'threshold' are considered signal
    (set to 0), a window of size 'window_size' is drawn around the signal
    points and those observations are also killed.
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
    """Compute noise temporal and spatial covariance.
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
    Randomly search noise snippets of 'temporal_size'.
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


def split_data(data, num_train, num_val, num_test):
    """Split data into train, val, and test sets.
    Parameters
    ----------
    data: numpy.ndarray
        spike data that is formatted in shape (num_spikes, n_chans, T)
    num_train:
        amount of data for the training set
    num_val: int
        amount of data for the validation set
    num_test: int
        amount of data for the test set
    Returns
    -------
    train_set: numpy.ndarray
    val_set: numpy.ndarray
    test_set: numpy.ndarray
    """
    train_set = []
    val_set = []
    test_set = []
    tot_num = num_train + num_val + num_test
    n_div = int(len(data) / tot_num)
    for i in range(n_div):
        #(N, train + test, num_chans_extract, 121)
        start = (num_train+num_val+num_test)*i
        train_set.append(data[start:start+num_train])
        val_set.append(data[start+num_train:start+num_train+num_val])
        test_set.append(data[start+num_train+num_val:start+num_train+num_val+num_test])
        
    return np.concatenate(train_set), np.concatenate(val_set), np.concatenate(test_set)


def pad_channels(wf, geoms, mc_start, mc_end, n_chans, spike_length_samples=121):
    """Pad extracted waveform data.
    Parameters
    ----------
    wf: numpy.ndarray
        spike that has shape (n_chans, spike_length_samples)
    geoms: numpy.ndarray
        probe geometry locations for wf. has shape (n_chans, (2/3)) for x, y(, z) locations of each channel
    mc_start: int
        start channel number
    mc_end: int
        end channel number in contiguous channel extraction
    n_chans: int
        number of channels in the data
    spike_length_samples: int
        number of samples for each waveform
    Returns
    -------
    zero padded versions of each of the wf, geom, and channel numbers, with a mask for each padded channel
    """
    curr_n_chans = mc_end - mc_start
    pad_beg = n_chans - curr_n_chans if mc_start == 0 else 0
    pad_end = n_chans - curr_n_chans if mc_start > 0 else 0
    
    wf_len = wf.shape[0]
    
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


def normalize_wf(wf):
    """Normalize waveform for use in cell type experiments.
    Parameters
    ----------
    wf: numpy.ndarray
        spike that has shape (n_chans, spike_length_samples)
    Returns
    -------
    waveform normalized to the range [0, 1] to retain shape information, but not amplitude information
    """
    if len(wf.shape) == 1:
        _ = wf.shape
        n_chans = None
    else:
        n_chans, _ = wf.shape
    wf = wf.flatten()
    if np.max(np.abs(wf),axis=0) != 0:
        wf /= np.max(np.abs(wf),axis=0)
    wf = wf.reshape(n_chans, -1) if n_chans is not None else wf
    return wf


def shift_wf(wf, trough=42, post_peak=79, pre_peak=47, jitter=5, real=False):
    """Take a shifted window of a waveform such that it's trough is aligned to a given value
    Parameters
    ----------
    wf: numpy.ndarray
        spike that has shape (n_chans, spike_length_samples)
    trough: int
        sample number of the trough to which the window should be aligned
    post_peak: int
        sample number after the peak of the waveform
    pre_peak: int
        sample number before the peak of the waveform
    jitter: int
        number of samples to shift the window in either direction to find proper alignment
    Returns
    -------
    waveform normalized to the range [0, 1] to retain shape information, but not amplitude information
    """
    if len(wf.shape) > 1:
        max_chan = math.floor(len(wf) / 2)
        max_chan_wf = wf[max_chan]
    else:
        max_chan_wf = wf
    max_samp_time = np.argmax(np.abs(max_chan_wf)[pre_peak-jitter:pre_peak+jitter+1]) + pre_peak-jitter
    
    pre_trough_window = max_samp_time - trough
    post_trough_window = max_samp_time + post_peak
    if len(wf.shape) > 1:
        shifted_wf = wf[:, pre_trough_window:post_trough_window]
    else:
        shifted_wf = wf[pre_trough_window:post_trough_window]
        
    return shifted_wf


def save_sim_covs(rec_path, save_path, spike_length_samples=121):
    """Create and save temporal and spatial covariance matrices for simulated data.
    Parameters
    ----------
    rec_path: str
        absolute path location of the binary file that contains channel recordings
    save_path: str
        absolute path location to store covariance matrices
    spike_length_samples: int
        number of samples for each waveform
    """
    recgen = mr.load_recordings(rec_path, load_waveforms=False)
    
    rec = si.MEArecRecordingExtractor(rec_path)
    rec = si.bandpass_filter(rec, dtype='float32')
    rec = si.common_reference(rec, reference='global', operator='median')
    rec = si.zscore(rec)
    
    norm_chan_recording = rec.get_traces()
    
    spatial_cov, temporal_cov = noise_whitener(norm_chan_recording, spike_length_samples, 50)
    
    np.save(os.path.join(save_path, 'spatial_cov_example.npy'), spatial_cov)
    np.save(os.path.join(save_path, 'temporal_cov_example.npy'), temporal_cov)
    
    
def save_real_covs(rec_path, save_path, spike_length_samples=121):
    """Create and save temporal and spatial covariance matrices for real data.
    Parameters
    ----------
    rec_path: str
        absolute path location of the binary file that contains channel recordings
    save_path: str
        absolute path location to store covariance matrices
    spike_length_samples: int
        number of samples for each waveform
    """
    sr = spikeglx.Reader(rec_path)
    rec = sr.read(nsel=slice(None))[0][:, :-1]
    
    spatial_cov, temporal_cov = noise_whitener(rec, spike_length_samples, 50)
    
    np.save(os.path.join(save_path, '/spatial_cov_example.npy'), spatial_cov)
    np.save(os.path.join(save_path, '/temporal_cov_example.npy'), temporal_cov)


def download_IBL(pid, save_folder, t_window=[0, 500], overwrite=True):
    """Extract and format data from a specific IBL session.
    Parameters
    ----------
    pid: str
        pid for the IBL session/recording
    save_folder: str
        absolute path to destination folder for the recording files to be stored.
    t_window: int
        specific window of time of the full recording to download and format in s
    overwrite: bool
        If overwrite is True, the save_folder (if it exists) will be overwritten.
    Returns
    -------
    standardized_file: str
        absolute path location of binary file for IBL session
    metadata_file: str
        absolute path location of corresponding metadata file for IBL session
    """
    one = ONE()
    eid, probe = one.pid2eid(pid)
    band = 'ap' # either 'ap' or 'lf'

    sr = Streamer(pid=pid, one=one, remove_cached=False, typ=band)
    sr._download_raw_partial(first_chunk=t_window[0], last_chunk=t_window[1] - 1)
    print(sr.one.load_dataset(sr.eid, f'*.{band}.meta', collection=f"*{sr.pname}"))

    sr.file_bin = sr.target_dir / '_spikeglx_ephysData_g0_t0.imec0.ap.stream.cbin'

    folder = Path(save_folder)
    if not folder.exists():
        # If it doesn't exist, create it
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Folder '{folder}' created.")
    else:
        if overwrite:
            shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Folder '{folder}' overwritten.")
        else:
            print(str(folder) + " already exists and overwrite=False. skipping destriping.")
    
    binary = Path(sr.file_bin)
    standardized_file = folder / f"{binary.stem}.normalized.bin"
    # run destriping
    sr = spikeglx.Reader(binary)
    h = sr.geometry
    if not standardized_file.exists():
        print("running destriping")
        batch_size_secs = 1
        assert sr.rl > 80, "download window must be larger than 4"
        batch_intervals_secs = 50
        # scans the file at constant interval, with a demi batch starting offset
        nbatches = int(
            np.floor((sr.rl - batch_size_secs) / batch_intervals_secs - 0.5)
        )
        wrots = np.zeros((nbatches, sr.nc - sr.nsync, sr.nc - sr.nsync))
        for ibatch in trange(nbatches, desc="destripe batches"):
            ifirst = int(
                (ibatch + 0.5) * batch_intervals_secs * sr.fs
                + batch_intervals_secs
            )
            ilast = ifirst + int(batch_size_secs * sr.fs)
            sample = voltage.destripe(
                sr[ifirst:ilast, : -sr.nsync].T, fs=sr.fs, neuropixel_version=1
            )
            np.fill_diagonal(
                wrots[ibatch, :, :],
                1 / rms(sample) * sr.sample2volts[: -sr.nsync],
            )

        wrot = np.median(wrots, axis=0)
        voltage.decompress_destripe_cbin(
            sr.file_bin,
            h=h,
            wrot=wrot,
            output_file=standardized_file,
            dtype=np.float32,
            nc_out=sr.nc - sr.nsync,
        )
        # also copy the companion meta-data file
        metadata_file = standardized_file.parent.joinpath(
                f"{sr.file_meta_data.stem}.normalized.meta"
            )
        shutil.copy(
            sr.file_meta_data,
            metadata_file,
        )
        
        print("done with destriping")
    metadata_file = standardized_file.parent.joinpath(
                f"{sr.file_meta_data.stem}.normalized.meta"
            )
    return standardized_file, metadata_file
    

def extract_IBL(bin_fp, meta_fp, pid, t_window=[0, 1100], use_labels=True, sampling_frequency=30_000):
    """Extract and format data from a specific IBL session.
    Parameters
    ----------
    bin_fp: str
        file path to the binary file containing channel recordings
    meta_fp: str
        file path to the meta file for bin_fp
    pid: str
        pid for the IBL session/recording
    t_window: int
        specific window of time of the recording that bin_fp contains
    use_labels: bool
        whether to put the kilosort sorted labels of neural units in the spike index (false for extracted spikes data)
    sampling_frequency: int
        sampling frequency of the recording (default 30,000 for IBL recordings)
    Returns
    -------
    spike_index: numpy.ndarray
        spike index of shape (2, len(spike_train)) or (3, len(spike_train)) that contains the spike train, max amplitude channel
        of each spike in the spike train(, and the putative kilosort neural unit from which each spike originates)
    geom: numpy.ndarray
    channel_index: numpy.ndarray
    templates: numpy.ndarray
    """
    one = ONE()
    ba = AllenAtlas()

    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    
    geom = read_geom_from_meta(Path(meta_fp))
    spike_times = spikes['times']
    print(spike_times)
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
    templates, _ = snr_templates.get_templates(aligned_spike_train, geom, bin_fp, closest_channels, reducer=np.median, do_temporal_decrease=False)
    mcs = np.array([templates[unit_id].ptp(0).argmax(0) for unit_id in range(len(templates))])
    
    spike_index = aligned_spike_train
    if use_labels:
        spike_index = np.vstack([spike_index[:, 0], np.array([mcs[unit] for unit in aligned_spike_train[:, 1]]), spike_index[:, 1]])
    else:
        spike_index[:, 1] = np.array([mcs[unit] for unit in aligned_spike_train[:, 1]])
    
    return spike_index, geom, channel_index, templates
    
    
def extract_sim(rec_path, wfs_per_unit, use_labels=True, geom_dims=(1,2), trough_offset=42, spike_length_samples=131, random_seed=0):
    """Extract and format data from a simulated session.
    Parameters
    ----------
    rec_path: str
        file path to the channel recording
    wfs_per_unit: int
        number of waveforms to extract per unit
    use_labels: bool
        whether to put the neuron label in the spike index (false for extracted spikes data)
    geom_dims: tuple
        which dims to use from the probe geometry
    trough_offset: int
        which sample to place the trough of a waveform when extracting from the recording
    spike_length_samples: int
        number of samples for each waveform
    random_seed: int
        random seed for waveform extraction
    Returns
    -------
    spike_index: numpy.ndarray
        spike index of shape (2, len(spike_train)) or (3, len(spike_train)) that contains the spike train, max amplitude channel
        of each spike in the spike train(, and the putative kilosort neural unit from which each spike originates)
    geom: numpy.ndarray
    we: SpikeInterface waveform extractor object
    """   
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
    """Chunk up a large recording for dataset generation.
    Parameters
    ----------
    spike_index: numpy.ndarray
        spike index of shape (2, len(spike_train)) or (3, len(spike_train)) that contains the spike train, max amplitude channel
        of each spike in the spike train(, and the putative kilosort neural unit from which each spike originates)
    max_proc_len: int
        number of spikes for each chunk
    Returns
    -------
    chunks: numpy.ndarray
    """   
    spike_idx_len = spike_index.shape[1]
    n_chunks = spike_idx_len // max_proc_len + 1
    chunks = []
    for i in range(n_chunks):
        start = i * max_proc_len
        chunk = spike_index[:, start:start+max_proc_len]
        chunks.append(chunk)
    
    return chunks

def combine_datasets(data_folder_list, save_folder):
    """Combine multiple CEED datasets to make a larger dataset. Can be used to train a model from multiple recordings.
    Parameters
    ----------
    data_folder_list: list
        list of absolute path to folders containing CEED datasets to combine
    save_folder: str
        absolute path to destination folder for combined dataset.
    """   
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    files_to_fix = ['spikes_', 'labels_', 'channel_num_', 'channel_spike_locs_']
    ttv = ['train.npy', 'val.npy', 'test.npy']
    
    for i, file in enumerate(files_to_fix):
        for split in ttv:
            data_list = []
            num_unique = 0
            for data_folder in data_folder_list:
                data = np.load(os.path.join(data_folder, file+split), allow_pickle=True)
                if file == 'labels_':
                    data = num_unique + data
                    if len(np.unique(data)) > 0:
                        num_unique += max(np.unique(data))                        
                data_list.append(data)
            if file == 'labels_':
                combined_data = np.concatenate(data_list)
            else:
                combined_data = np.vstack(data_list)
            np.save(os.path.join(save_folder, file+split), combined_data)
    #add spatial and temporal covariances to separate folders
    for i, data_folder in enumerate(data_folder_list):
        temporal_cov = np.load(os.path.join(data_folder, 'temporal_cov.npy'), allow_pickle=True)
        spatial_cov = np.load(os.path.join(data_folder, 'temporal_cov.npy'), allow_pickle=True)
        cov_folder = f"/covariances_ds{i}"
        if not os.path.exists(save_folder+cov_folder):
            os.makedirs(save_folder+cov_folder)
        np.save(os.path.join(save_folder+cov_folder, 'temporal_cov.npy'), temporal_cov)
        np.save(os.path.join(save_folder+cov_folder, 'spatial_cov.npy'), spatial_cov)


def make_dataset(bin_path, spike_index, geom, save_path, we=None, 
                 templates=None, chan_index=None, num_chans_extract=21, unit_ids=None, 
                 save_covs=False, train_num=1200, val_num=0, test_num=200, plot=False, inference=False, normalize=False,
                 shift=False, save_fewer=False, random_seed=0):
    """Extract and format data from a simulated session.
    Parameters
    ----------
    bin_path: str
        file path to the channel recording
    spike_index: numpy.ndarray
        spike index of shape (2, len(spike_train)) or (3, len(spike_train)) that contains the spike train, max amplitude channel
        of each spike in the spike train(, and the putative kilosort neural unit from which each spike originates)
    geom: numpy.ndarray
        the probe geometry for the recording.
    save_path: str
        absolute location in which the dataset will be saved (dir will be created if nonexistent)
    we: SpikeInterface waveform extractor object
    templates: numpy.ndarray
        templates of each of the neurons in recording window
    chan_index: numpy.ndarray
    num_chans_extract: int
        the total number of neighboring channels to extract per waveform, centered on the max amplitude channel
    unit_ids: list
        selection of units for which waveforms will be extracted. 
    save_covs: bool
        whether to process and save out spatial and temporal noise covariance matrices for the recording
    train_num: int or tuple
        number of waveforms to put in training set (or (min, max) number of waveforms allowed per unit)
    val_num: int or tuple
        number of waveforms to put in validation set (or (min, max) number of waveforms allowed per unit)
    test_num: int or tuple
        number of waveforms to put in test set (or (min, max) number of waveforms allowed per unit)
    plot: bool
        save out plot of spikes per unit in a plot folder 
    inference: bool
        flag to only save out test sets of spikes for inference, rather than train/val sets as well for training new models
    normalize: bool
        flag to normalize spikes into range [-1, 1], especially useful for celltype tasks/training
    shift: bool
        flag to shift extracted spikes to get proper alignment of trough
    save_fewer: bool
        flag to save out neural units spikes even if there are not enough to put in requested amounts for train, val, and test sets
    random_seed: int
        random seed for waveform extraction
    """   

    np.random.seed(random_seed)
    num_waveforms = train_num + val_num + test_num
    spikes_array = []
    geom_locs_array = []
    max_chan_array = []
    masks_array = []
    chosen_units = []
    labels_array = []
    spike_frames_templates = []
    curr_row = 0
    max_proc_len = 25000
    num_template_amps_shift = 4
    spike_length_to_extract = 131 if shift else 121
    spike_length_samples = 121
    num_chans = math.floor(num_chans_extract/2)
    tot_num_chans = geom.shape[0]
    if we is not None:
        depth_order = np.argsort(geom[:,2])
        geom = geom[depth_order]
    num_waveforms = train_num + val_num + test_num
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 22
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if save_covs:
        if we is not None:
            save_sim_covs(bin_path, save_path)
        else:
            save_real_covs(bin_path, save_path)
    else:
        dir_path = os.path.abspath(os.path.dirname(__file__))
        print(f"copying over noise covariances from previous recording from {dir_path}/noise_covariances_IBL_dy016")
        spatial_cov = np.load(dir_path + '/noise_covariances_IBL_dy016/spatial_cov.npy')
        temporal_cov = np.load(dir_path + '/noise_covariances_IBL_dy016/temporal_cov.npy')
        np.save(os.path.join(save_path, 'spatial_cov.npy'), spatial_cov)
        np.save(os.path.join(save_path, 'temporal_cov.npy'), temporal_cov)
        
    if unit_ids is None:
        data_chunks = chunk_data(spike_index, max_proc_len)
        if spike_index.shape[0] == 3:
            chosen_units = np.unique(spike_index[2, :])
        for i in range(len(data_chunks)):
            curr_chunk = data_chunks[i]
            waveforms, _ = read_waveforms(curr_chunk[0, :], bin_path, 
                                          n_channels=geom.shape[0], spike_length_samples=spike_length_to_extract)
            mcs = curr_chunk[1, :]
            for i, waveform in enumerate(waveforms):
                mc_curr = mcs[i]
                mc_start = max(mc_curr - num_chans, 0)
                mc_end = min(mc_curr + num_chans + 1, tot_num_chans)
                crop_wf, crop_chan, crop_geom, mask = pad_channels(waveform.T[mc_start:mc_end], 
                                                                   geom[mc_start:mc_end], mc_start, mc_end, 
                                                                   num_chans_extract)
                shifted_wf = shift_wf(crop_wf) if shift else crop_wf
                spikes_array.append(shifted_wf)
                geom_locs_array.append(crop_geom)
                max_chan_array.append(crop_chan) 
    else:
        spike_index = spike_index.T
        for k, unit_id in enumerate(unit_ids):
            curr_temp_wfs = []
            curr_geom_locs = []
            curr_spike_max_chan = []
            curr_masks = []
            curr_labels = []
            
            if not save_fewer and len(spike_index[:,0][np.where(spike_index[:,2]==unit_id)[0]]) < num_waveforms:
                print("Unit {} only has {} spikes, but {} are requested. Skipping extraction for this unit...".format(unit_id, 
                                                                len(spike_index[:,0][np.where(spike_index[:,2]==unit_id)[0]]), num_waveforms))
                continue
            chosen_units.append(unit_id)

            # Extract from Simulated Data extractor, otherwise use real data extraction
            if we is not None:
                waveforms = we.get_waveforms("#{}".format(str(unit_id)))
                waveforms = waveforms[:num_waveforms, :, depth_order]
                templates = we.get_all_templates()
                templates = templates[:, :, depth_order]
            else:
                min_num_spikes = min(len(spike_index[:,0][np.where(spike_index[:,2]==unit_id)[0]]), num_waveforms)
                if min_num_spikes == num_waveforms:
                    min_num_spikes += min(600, len(spike_index[:,0][np.where(spike_index[:,2]==unit_id)[0]]) - num_waveforms)
                spike_frames_template = np.random.choice(spike_index[:,0][np.where(spike_index[:,2] == unit_id)[0]], 
                                                            size=min_num_spikes, replace=False)
                waveforms, _ = read_waveforms(spike_frames_template, bin_path, channel_index=chan_index,
                                                  n_channels=geom.shape[0], spike_length_samples=spike_length_to_extract)
                waveforms = waveforms[:num_waveforms]

            mc = np.unique(spike_index[np.where(spike_index[:, 2] == unit_id)[0], 1])[0]
            mc = templates[unit_id].ptp(0).argmax()
            mcs_template = templates[unit_id].ptp(0).argsort()[::-1][:num_template_amps_shift]
            shifts = np.abs(waveforms[:,:,mcs_template]).max(1).argmax(1)
            mcs_shifted = mcs_template[shifts]

            for i, waveform in enumerate(waveforms):
                mc_curr = mcs_shifted[i]
                mc_start = max(mc_curr - num_chans, 0)
                mc_end = min(mc_curr + num_chans + 1, tot_num_chans)
                curr_wf = normalize_wf(waveform.T[mc_start:mc_end]) if normalize else waveform.T[mc_start:mc_end]
                curr_geom = geom[mc_start:mc_end, 1:] if geom.shape[1] == 3 else geom[mc_start:mc_end]
                crop_wf, crop_chan, crop_geom, mask = pad_channels(curr_wf,
                                                                   curr_geom, mc_start, mc_end, 
                                                                   num_chans_extract)
                
                shifted_wf = shift_wf(crop_wf) if shift else crop_wf
                
                curr_temp_wfs.append(shifted_wf)
                curr_geom_locs.append(crop_geom)
                curr_spike_max_chan.append(crop_chan)
                curr_masks.append(mask)
                curr_labels.append(unit_id)

            spikes_array.append(np.asarray(curr_temp_wfs))
            geom_locs_array.append(np.asarray(curr_geom_locs))
            max_chan_array.append(np.asarray(curr_spike_max_chan))
            masks_array.append(np.asarray(curr_masks))
            labels_array.append(np.asarray(curr_labels))

            if plot:
                plot_folder = os.path.join(save_path, "wf_plots")
                fig = plt.figure(figsize=(2 + int(num_chans_extract), 3))
                gs = GridSpec(1, num_chans_extract+1, figure=fig)
                ax0 = fig.add_subplot(gs[0])
                ax0.title.set_text('Unit {}'.format(str(unit_id)))
                curr_temp = normalize_wf(templates[unit_id].T[mc]) if normalize else templates[unit_id].T[mc]
                curr_temp = shift_wf(curr_temp) if we is not None else curr_temp
                ax0.plot(x, curr_temp)
                ax0.axvline(42)
                ax0.axes.get_xaxis().set_visible(False)
                ax1 = fig.add_subplot(gs[1:], sharey=ax0)
                for waveform in curr_temp_wfs[:100]:
                    ax1.plot(waveform.flatten(), color='blue', alpha=.1)
                    curr_temp = normalize_wf(templates[unit_id].T[mc_start:mc_end]) if normalize else templates[unit_id].T[mc_start:mc_end]
                    curr_temp = shift_wf(curr_temp).flatten() if we is not None else curr_temp.flatten()
                    ax1.plot(curr_temp, color='red')
                    vlines = 42 + (121*np.arange(num_chans_extract))
                    for vline in vlines:
                        ax1.axvline(vline)
                    ax1.axes.get_yaxis().set_visible(False)
                    ax1.axes.get_xaxis().set_visible(False)
                curr_row += 1
                fig.subplots_adjust(wspace=0, hspace=0.25)
                plt.savefig(os.path.join(plot_folder, f'unit{str(unit_id)}'))
                plt.close()
    if not save_fewer:
        spikes_array = np.concatenate(spikes_array)
        geom_locs_array = np.concatenate(geom_locs_array)
        max_chan_array = np.concatenate(max_chan_array)
        masks_array = np.concatenate(masks_array)
        if unit_ids is not None:
            labels_array = np.concatenate(labels_array)
    if not inference:
        if not save_fewer:
            print("making train, val, test splits")
            train_set, val_set, test_set = split_data(spikes_array, train_num, val_num, test_num, )
            train_geom_locs, val_geom_locs, test_geom_locs = split_data(geom_locs_array, train_num, val_num, test_num)
            train_max_chan, val_max_chan, test_max_chan = split_data(max_chan_array, train_num, val_num, test_num)
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
            np.save(os.path.join(save_path, 'selected_units.npy'), np.array(chosen_units))

            if unit_ids is not None:
                train_labels, val_labels, test_labels = split_data(labels_array, train_num, val_num, test_num)
                np.save(os.path.join(save_path, 'labels_train.npy'), train_labels)
                np.save(os.path.join(save_path, 'labels_val.npy'), val_labels)
                np.save(os.path.join(save_path, 'labels_test.npy'), test_labels)
        else:
            # save out all spikes (even for neurons that didn't have enough to extract) into train and test sets.
            print("saving no split train dataset")
            train_set = np.concatenate([curr_arr[:train_num] for curr_arr in spikes_array])
            val_set = np.concatenate([curr_arr[train_num:train_num+val_num] for curr_arr in spikes_array])
            test_set = np.concatenate([curr_arr[train_num+val_num:train_num+val_num+test_num] for curr_arr in spikes_array])
            train_geom_locs = np.concatenate([curr_arr[:train_num] for curr_arr in geom_locs_array])
            val_geom_locs = np.concatenate([curr_arr[train_num:train_num+val_num] for curr_arr in geom_locs_array])
            test_geom_locs = np.concatenate([curr_arr[train_num+val_num:train_num+val_num+test_num] for curr_arr in geom_locs_array])
            train_max_chan = np.concatenate([curr_arr[:train_num] for curr_arr in max_chan_array])
            val_max_chan = np.concatenate([curr_arr[train_num:train_num+val_num] for curr_arr in max_chan_array])
            test_max_chan = np.concatenate([curr_arr[train_num+val_num:train_num+val_num+test_num] for curr_arr in max_chan_array])
            np.save(os.path.join(save_path, 'spikes_test.npy'), test_set)
            np.save(os.path.join(save_path, 'spikes_val.npy'), val_set)
            np.save(os.path.join(save_path, 'spikes_train.npy'), train_set)
            np.save(os.path.join(save_path, 'channel_spike_locs_test.npy'), test_geom_locs)
            np.save(os.path.join(save_path, 'channel_spike_locs_val.npy'), val_geom_locs)
            np.save(os.path.join(save_path, 'channel_spike_locs_train.npy'), train_geom_locs)
            np.save(os.path.join(save_path, 'channel_num_test.npy'), test_max_chan)
            np.save(os.path.join(save_path, 'channel_num_val.npy'), val_max_chan)
            np.save(os.path.join(save_path, 'channel_num_train.npy'), train_max_chan)
            
            if unit_ids is not None:
                train_labs = np.concatenate([curr_arr[:train_num] for curr_arr in labels_array])
                val_labs = np.concatenate([curr_arr[train_num:train_num+val_num] for curr_arr in labels_array])
                test_labs = np.concatenate([curr_arr[train_num+val_num:train_num+val_num+test_num] for curr_arr in labels_array])
                np.save(os.path.join(save_path, 'labels_test.npy'), test_labs)
                np.save(os.path.join(save_path, 'labels_val.npy'), val_labs)
                np.save(os.path.join(save_path, 'labels_train.npy'), train_labs)
            
        return train_set, val_set, test_set, train_geom_locs, val_geom_locs, test_geom_locs, train_max_chan, val_max_chan, test_max_chan
    else:
        #save out full dataset with no splits (useful for inference)
        if not save_fewer:
            print("saving no split results")
            test_set = spikes_array
            test_geom_locs = geom_locs_array
            test_max_chan = max_chan_array
            np.save(os.path.join(save_path, 'spikes_test.npy'), test_set)
            np.save(os.path.join(save_path, 'channel_spike_locs_test.npy'), test_geom_locs)
            np.save(os.path.join(save_path, 'channel_num_test.npy'), test_max_chan)

            np.save(os.path.join(save_path, 'geom.npy'), geom)
            np.save(os.path.join(save_path, 'selected_units.npy'), np.array(chosen_units))

            if unit_ids is not None:
                np.save(os.path.join(save_path, 'labels_test.npy'), labels_array)
        else:
            test_set = np.concatenate([curr_arr for curr_arr in spikes_array])
            test_geom_locs = np.concatenate([curr_arr for curr_arr in geom_locs_array])
            test_max_chan = np.concatenate([curr_arr for curr_arr in max_chan_array])
            np.save(os.path.join(save_path, 'spikes_test.npy'), test_set)
            np.save(os.path.join(save_path, 'channel_spike_locs_test.npy'), test_geom_locs)
            np.save(os.path.join(save_path, 'channel_num_test.npy'), test_max_chan)
            
            if unit_ids is not None:
                test_labs = np.concatenate([curr_arr for curr_arr in labels_array])
                np.save(os.path.join(save_path, 'labels_test.npy'), test_labs)

        return test_set, test_geom_locs, test_max_chan
    