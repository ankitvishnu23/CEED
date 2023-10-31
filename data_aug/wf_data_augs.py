from __future__ import print_function, division
import os
import math
import numpy as np
from sklearn.decomposition import PCA

import torch
from scipy import signal

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

import glob

class AmpJitter(object):
    """Rescales waveform amplitude by some small factor"""

    def __init__(self, lo=0.9, hi=1.1):
        """
        Args:
            lo: float
                the low end of the amplitude distortion.
            hi: float
                the high end of the amplitude distortion.
        """
        self.lo = lo
        self.hi = hi

    def __call__(self, sample):
        chan_locs = None
        if len(sample) == 2:
            wf, chan_nums = sample
        elif len(sample) == 3:
            wf, chan_nums, chan_locs = sample
        else:
            wf = sample

        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        n_chans = wf.shape[0]

        # randomly select amp jitter scaling value and apply to waveform in each channel
        amp_jit_value = np.random.uniform(self.lo, self.hi)
        amp_jit = np.array([amp_jit_value for i in range(n_chans)])

        wf = wf * amp_jit[:, None]

        if chan_locs is None:
            return [wf, chan_nums]

        return [wf, chan_nums, chan_locs]


class GaussianNoise(object):
    """Adds Gaussian noise to each waveform"""

    def __call__(self, sample):
        chan_locs = None
        if len(sample) == 2:
            wf, chan_nums = sample
        elif len(sample) == 3:
            wf, chan_nums, chan_locs = sample
        else:
            wf = sample

        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        n_chans = wf.shape[0]
        w = wf.shape[1]

        noise_wf = np.random.normal(0, 1, (n_chans, w))
        wf = np.add(wf, noise_wf)

        return [wf, chan_nums]


class SmartNoise(object):
    """Add randomly generated noise from distribution determined by spatial and temporal covariances"""

    temporal_name = "temporal_cov.npy"
    spatial_name = "spatial_cov.npy"
    temporal_name = "temporal_cov.npy"
    spatial_name = "spatial_cov.npy"

    def __init__(
        self,
        root_folder,
        temporal_cov=None,
        spatial_cov=None,
        noise_scale=1.0,
        normalize=False,
    ):
        """
        Args:
            root_folder: str/Path
                location at which to retrieve the covariance matrices.
            temporal_cov: str
                optional. temporal covariance matrix.
            spatial_cov: str
                optional. spatial covariance matrix.
            noise_scale: float
                scale the randomly generated noise by this amount.
            normalize: bool
                normalize the noise to [0, 1] if True. used for cell type classification model training.
        """
        self.root_folder = root_folder
        if temporal_cov is None:
            temporal_cov = np.load(os.path.join(self.root_folder, self.temporal_name))
        if spatial_cov is None:
            spatial_cov = np.load(os.path.join(self.root_folder, self.spatial_name))
        self.temporal_cov = temporal_cov
        self.spatial_cov = spatial_cov
        # self.noise_scale = np.float64(noise_scale)
        self.noise_scale = np.float32(noise_scale)
        self.normalize = normalize

    def __call__(self, sample):
        chan_locs = None
        if len(sample) == 2:
            wf, chan_nums = sample
        elif len(sample) == 3:
            wf, chan_nums, chan_locs = sample
        else:
            wf = sample

        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        n_chans = wf.shape[0]
        w = wf.shape[1]

        assert self.temporal_cov.shape[0] == w

        n_chans_total, _ = self.spatial_cov.shape
        waveform_length, _ = self.temporal_cov.shape

        if type(chan_nums) != np.int64:
            chan_nums[chan_nums > n_chans_total - 1] = n_chans_total - 1
            chan_nums[chan_nums < 0] = 0
        chan_nums = chan_nums.astype(int)  # sometimes chan_nums is a float

        noise = np.random.normal(size=(waveform_length, n_chans_total))

        noise = np.reshape(
            np.matmul(noise, self.spatial_cov), (waveform_length, n_chans_total)
        )[:, chan_nums]

        the_noise = np.reshape(np.matmul(noise.T, self.temporal_cov).T, (-1, n_chans))

        noise_to_add = the_noise.T
        noise_to_add = (
            self.normalize_wf(noise_to_add) if self.normalize else noise_to_add
        )

        noise_wfs = self.noise_scale * noise_to_add
        wf = wf + noise_wfs

        if chan_locs is None:
            return [wf, chan_nums]

        return [wf, chan_nums, chan_locs]

    def normalize_wf(self, wf):
        if len(wf.shape) == 1:
            _ = wf.shape
            n_chans = None
        else:
            n_chans, _ = wf.shape
        wf = wf.flatten()
        wf /= np.max(np.abs(wf), axis=0)
        wf = wf.reshape(n_chans, -1) if n_chans is not None else wf
        return wf


class TorchSmartNoise(object):
    """Add randomly generated noise from distribution determined by spatial and temporal covariances.
    Generation and addition happens on gpu for speed purposes.
    """

    temporal_covs_path = "/covariances/temporal*"
    spatial_covs_path = "/covariances/spatial*"

    def __init__(
        self,
        root_folder=None,
        temporal_cov=None,
        spatial_cov=None,
        noise_scale=1.0,
        normalize=False,
        gpu=0,
        p=0.5,
    ):
        """
        Args:
            root_folder: str/Path
                location at which to retrieve the covariance matrices.
            temporal_cov: str
                optional. temporal covariance matrix.
            spatial_cov: str
                optional. spatial covariance matrix.
            noise_scale: float
                scale the randomly generated noise by this amount.
            normalize: bool
                normalize the noise to [0, 1] if True. used for cell type classification model training.
            gpu: int
                gpu num on which to perform torch noise augmentation.
            p: float
                noise will be generated and added to waveform with this probability.
        """
        self.root_folder = root_folder
        if temporal_cov is None or spatial_cov is None:
            temporal_covs_files = glob.glob(str(root_folder) + self.temporal_covs_path)
            spatial_covs_files = glob.glob(str(root_folder) + self.spatial_covs_path)
            spatial_temporal_covs = [
                (spatial_covs_files[i], temporal_covs_files[i])
                for i in range(len(temporal_covs_files))
            ]
        else:
            spatial_temporal_covs = [(spatial_cov, temporal_cov)]
        self.spatial_temporal_covs_tensors = []
        for st_cov_tuple in spatial_temporal_covs:
            spatial_cov_path, temporal_cov_path = st_cov_tuple
            spatial_cov = np.load(spatial_cov_path)
            temporal_cov = np.load(temporal_cov_path)
            spatial_cov_tensor = (
                torch.from_numpy(spatial_cov).cuda(gpu, non_blocking=True).float()
            )
            temporal_cov_tensor = (
                torch.from_numpy(temporal_cov).cuda(gpu, non_blocking=True).float()
            )
            self.spatial_temporal_covs_tensors.append(
                (spatial_cov_tensor, temporal_cov_tensor)
            )
        self.noise_scale = np.float32(noise_scale)
        self.normalize = normalize
        self.gpu = gpu
        self.prob = p

    def __call__(self, sample):
        # sample noise from random recording
        rec_id = np.random.choice(range(len(self.spatial_temporal_covs_tensors)))
        spatial_cov, temporal_cov = self.spatial_temporal_covs_tensors[rec_id]

        chan_locs = None
        if len(sample) == 2:
            wf, chan_nums = sample
            # wf, chan_nums = sample
        elif len(sample) == 3:
            wf, chan_nums, chan_locs = sample
            # wf, chan_nums, chan_locs = sample
        else:
            wf = sample
        if self.prob > torch.rand(1):
            if len(wf.shape) == 2:
                wf = np.expand_dims(wf, axis=1)
            bs, n_chans, w = wf.shape

            assert temporal_cov.shape[0] == w

            n_chans_total, _ = spatial_cov.shape
            wf_length, _ = temporal_cov.shape

            # pad channels that cross edges with edge chan numbers
            if type(chan_nums) != np.int64:
                chan_nums[chan_nums > n_chans_total - 1] = n_chans_total - 1
                chan_nums[chan_nums < 0] = 0
            chan_nums = chan_nums.astype(int)  # sometimes chan_nums is a float

            with torch.no_grad():
                noise = torch.cuda.FloatTensor(bs, wf_length, n_chans_total).normal_()

                spatial_adj_noise = torch.bmm(
                    noise, spatial_cov.expand(bs, -1, -1)
                ).view((bs, wf_length, n_chans_total))
                spatial_adj_noise = torch.stack(
                    [
                        spatial_adj_noise[i, :, chan_num]
                        for i, chan_num in enumerate(chan_nums)
                    ]
                )

                the_noise = torch.bmm(
                    spatial_adj_noise.permute(0, 2, 1), temporal_cov.expand(bs, -1, -1)
                ).view(bs, n_chans, -1)

                noise_wfs = self.noise_scale * the_noise
                wf = wf + noise_wfs

        return wf

    def normalize_wf(self, wf):
        if len(wf.shape) == 1:
            _ = wf.shape
            n_chans = None
        else:
            n_chans, _ = wf.shape
        wf = wf.flatten()
        wf /= np.max(np.abs(wf), axis=0)
        wf = wf.reshape(n_chans, -1) if n_chans is not None else wf
        return wf


class Collide(object):
    """Select a waveform from the training set, scale it, offset it, and add it in order to simulate a spiking collision"""

    spikes_fn = "spikes_train.npy"

    def __init__(self, root_folder, spikes=None):
        """
        Args:
            root_folder: str/Path
                location at which to retrieve the spikes file.
            spikes: numpy.arraylike
                optional. training set of spikes.
        """
        self.root_folder = root_folder
        if spikes is None:
            spikes = np.load(os.path.join(self.root_folder, self.spikes_fn))
        self.spikes = spikes

    def __call__(self, sample):
        chan_locs = None
        if len(sample) == 2:
            wf, chan_nums = sample
        elif len(sample) == 3:
            wf, chan_nums, chan_locs = sample
        else:
            wf = sample

        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)

        temp_idx = np.random.randint(0, len(self.spikes))
        temp_sel = self.spikes[temp_idx]
        temp_sel = (
            np.expand_dims(temp_sel, axis=0) if len(temp_sel.shape) == 1 else temp_sel
        )

        scale = np.random.uniform(0.2, 1)
        shift = (2 * np.random.binomial(1, 0.5) - 1) * np.random.randint(5, 60)

        temp_sel = temp_sel * scale
        temp_sel = self.shift_chans(temp_sel, shift)

        wf = np.add(wf, temp_sel)

        if chan_locs is None:
            return [wf, chan_nums]

        return [wf, chan_nums, chan_locs]

    def shift_chans(self, wf, shift_):
        # use template feat_channel shifts to interpolate shift of all spikes on all other chans
        int_shift = int(math.ceil(shift_)) if shift_ >= 0 else -int(math.floor(shift_))
        curr_wf_pos = np.pad(wf, ((0, 0), (0, int_shift)), "constant")
        curr_wf_neg = np.pad(wf, ((0, 0), (int_shift, 0)), "constant")
        if int(shift_) == shift_:
            ceil = int(shift_)
            temp = (
                np.roll(curr_wf_pos, ceil, axis=1)[:, :-int_shift]
                if shift_ > 0
                else np.roll(curr_wf_neg, ceil, axis=1)[:, int_shift:]
            )
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            if shift_ > 0:
                temp = (np.roll(curr_wf_pos, ceil, axis=1) * (shift_ - floor))[
                    :, :-ceil
                ] + (np.roll(curr_wf_pos, floor, axis=1) * (ceil - shift_))[:, :-ceil]
            else:
                temp = (np.roll(curr_wf_neg, ceil, axis=1) * (shift_ - floor))[
                    :, -floor:
                ] + (np.roll(curr_wf_neg, floor, axis=1) * (ceil - shift_))[:, -floor:]
        wf_final = temp

        return wf_final


class Jitter(object):
    """Temporally jitter the waveform through resampling"""

    def __init__(self, up_factor=8, sample_rate=30000, shift=2):
        """
        Args:
            up_factor: int
                resample the waveform by this amount.
            sample_rate: int
                sampling rate of the recording.
            shift: int
                shift the jittered waveform forwards/backwards this number of samples.
        """
        assert isinstance(up_factor, (int))
        assert isinstance(sample_rate, (int))
        self.up_factor = up_factor
        self.sample_rate = sample_rate
        self.shift = shift

    def __call__(self, sample):
        chan_locs = None
        if len(sample) == 2:
            wf, chan_nums = sample
        elif len(sample) == 3:
            wf, chan_nums, chan_locs = sample
        else:
            wf = sample

        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        n_chans = wf.shape[0]
        w = wf.shape[1]

        up_temp = signal.resample(x=wf, num=w * self.up_factor, axis=1)

        idx = np.arange(0, w)[:, None] * self.up_factor + np.arange(self.up_factor)
        up_shifted_temp = np.transpose(up_temp[:, idx], (0, 2, 1))

        shift = (2 * np.random.binomial(1, 0.5) - 1) * np.random.uniform(0, self.shift)

        idx_selection = np.random.choice(self.up_factor)
        idxs = np.array([idx_selection for i in range(n_chans)])
        wf = up_shifted_temp[np.arange(n_chans), idxs]
        wf = self.shift_chans(wf, shift)

        if chan_locs is None:
            return [wf, chan_nums]

        return [wf, chan_nums, chan_locs]

    def shift_chans(self, wf, shift_):
        # use template feat_channel shifts to interpolate shift of all spikes on all other chans
        int_shift = int(math.ceil(shift_)) if shift_ >= 0 else -int(math.floor(shift_))
        curr_wf_pos = np.pad(wf, ((0, 0), (0, int_shift)), "constant")
        curr_wf_neg = np.pad(wf, ((0, 0), (int_shift, 0)), "constant")
        if int(shift_) == shift_:
            ceil = int(shift_)
            temp = (
                np.roll(curr_wf_pos, ceil, axis=1)[:, :-int_shift]
                if shift_ > 0
                else np.roll(curr_wf_neg, ceil, axis=1)[:, int_shift:]
            )
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            if shift_ > 0:
                temp = (np.roll(curr_wf_pos, ceil, axis=1) * (shift_ - floor))[
                    :, :-ceil
                ] + (np.roll(curr_wf_pos, floor, axis=1) * (ceil - shift_))[:, :-ceil]
            else:
                temp = (np.roll(curr_wf_neg, ceil, axis=1) * (shift_ - floor))[
                    :, -floor:
                ] + (np.roll(curr_wf_neg, floor, axis=1) * (ceil - shift_))[:, -floor:]
        wf_final = temp

        return wf_final


class Crop(object):
    """Crop a subset of channels from the waveform"""

    def __init__(self, prob=0.5, num_extra_chans=2, ignore_chan_num=False):
        """
        Args:
            prob: float
                crop will be applied to waveform with this probability.
            num_extra_chans: int
                number of channels on each side of max channel to use for crop window.
                total number of channels in cropped wf will be (2 * num_extra_chans + 1)
            ignore_chan_num: bool
                whether to return channel numbers and locations if they are present.
        """
        self.prob = prob
        self.num_extra_chans = num_extra_chans
        self.ignore_chan_num = ignore_chan_num
    def __call__(self, sample):
        if len(sample) == 3:
            wf, chan_nums, chan_locs = sample
        elif len(sample) == 2:
            wf, chan_nums = sample
            chan_locs = None
        else:
            wf = sample
            chan_nums = None
            chan_locs = None

        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        n_chans = wf.shape[0]

        max_chan_ind = math.floor(n_chans / 2)

        apply = np.random.binomial(1, self.prob)
        if apply:
            shift = np.random.randint(-self.num_extra_chans, self.num_extra_chans + 1)
            max_chan_ind += shift
        wf = wf[
            max_chan_ind
            - self.num_extra_chans : max_chan_ind
            + self.num_extra_chans
            + 1
        ]

        if not self.ignore_chan_num:
            if type(chan_nums) != np.int64:
                chan_nums = chan_nums[
                    max_chan_ind
                    - self.num_extra_chans : max_chan_ind
                    + self.num_extra_chans
                    + 1
                ]

        if chan_locs is not None:
            # verify that channel location is not for a single channel waveform
            if len(chan_locs.shape) > 1 and chan_locs.shape[0] != 1:
                chan_locs = chan_locs[
                    max_chan_ind
                    - self.num_extra_chans : max_chan_ind
                    + self.num_extra_chans
                    + 1
                ]

        # in single channel case the wf will become 1 dimensional
        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        if self.ignore_chan_num:
            if chan_locs is not None:
                return wf.astype("float32"), chan_locs
            return wf.astype("float32")

        if chan_locs is None:
            return [wf, chan_nums]

        return [wf, chan_nums, chan_locs]


class PCA_Reproj(object):
    """Fits PCA on set of spikes and reprojects a spike through PCA object"""

    spikes_file = "spikes_train.npy"

    def __init__(self, root_folder=None, spikes_file=None, pca_dim=5):
        """
        Args:
            root_folder: str/Path
                location at which to retrieve the spikes file.
            spikes_file: str
                file containing set of spikes used, on which PCA will be fit.
            pca_dim: int
                number of dimensions to be kept with PCA and which will be used to reconstruct wf.
        """
        assert isinstance(pca_dim, (int))
        if root_folder is not None:
            self.root_folder = root_folder
        if spikes_file is not None:
            self.spikes_file = spikes_file
        self.spikes = np.load(os.path.join(self.root_folder, self.spikes_file))
        self.pca_dim = pca_dim
        self.pca_ = PCA(n_components=self.pca_dim).fit(self.spikes)
        self.spikes_mean = np.mean(self.spikes, axis=0)

    def __call__(self, sample):
        transform = self.pca_.transform(sample.reshape(1, -1))
        recon = self.pca_.inverse_transform(transform)[0]

        return recon


class ElectrodeDropout(object):
    """Zero out a channel to mimic the electrode breaking."""

    def __init__(self, prob=0.1):
        """
        Args:
            prob: float
                each channel will be dropped with this probability.
        """
        self.p_drop_chan = prob

    def __call__(self, wf):
        n_chan, n_times = wf.shape
        chan_mask = -1 * np.random.binomial(1, self.p_drop_chan, n_chan) + 1

        wf[chan_mask == 0] = np.zeros(n_times)
        return wf


class ToWfTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if len(sample) == 2:
            wf, chan_nums = sample
        elif len(sample) == 3:
            wf, chan_nums, chan_locs = sample
        else:
            wf = sample

        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)

        if len(sample) == 3:
            return torch.from_numpy(wf.astype("float32")), chan_locs
        elif len(sample) == 2:
            return torch.from_numpy(wf.astype("float32"))

        return torch.from_numpy(wf.astype("float32"))


# For integration with TorchSmartNoise
class TorchToWfTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if len(sample) == 2:
            wf, chan_nums = sample
        elif len(sample) == 3:
            wf, chan_nums, chan_locs = sample
        else:
            wf = sample

        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)

        if len(sample) == 3:
            return torch.from_numpy(wf.astype("float32")), chan_nums, chan_locs
        elif len(sample) == 2:
            return torch.from_numpy(wf.astype("float32")), chan_nums

        return torch.from_numpy(wf.astype("float32"))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.ion()  # interactive mode
