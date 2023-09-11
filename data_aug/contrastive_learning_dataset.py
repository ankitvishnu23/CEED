from fileinput import filename
import torch
import numpy as np
import os

from torchvision.transforms import transforms
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from data_aug.view_generator import ContrastiveLearningViewGenerator, LabelViewGenerator
# from exceptions.exceptions import InvalidDatasetSelection
from data_aug.wf_data_augs import AmpJitter, Jitter, Collide, SmartNoise, ToWfTensor, PCA_Reproj, Crop, TorchToWfTensor
from typing import Any, Callable, Optional, Tuple

class WFDataset(Dataset):
    train_set_fn = "spikes_train.npy"
    spike_mcs_fn = "channel_num_train.npy"
    chan_coords_fn = "channel_spike_locs_train.npy"
    targets_fn = "labels_train.npy"
    # masks_fn = "masks_train.npy"

    def __init__(
        self,
        root: str,
        use_chan_pos: bool = False, 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        detected_spikes: bool = False
    ) -> None:

        super().__init__()

        self.data: Any = []

        # now load the numpy array
        self.data = np.load(os.path.join(root, self.train_set_fn))
        print(self.data.shape)
        self.root = root
        self.max_chans = np.load(os.path.join(root, self.spike_mcs_fn))
        self.transform = transform
        self.detected_spikes = detected_spikes

        if detected_spikes:
            self.targets = None
            self.target_transform = None
        else:
            self.targets = np.load(os.path.join(root, self.targets_fn))
            self.target_transform = target_transform
        self.channel_locs = np.load(os.path.join(root, self.chan_coords_fn))
        self.use_chan_pos = use_chan_pos
        # self.masks = np.load(os.path.join(root, self.masks_fn))


    def __getitem__(self, index: int) -> Any :
        """
        Args:
            index (int): Index
        Returns:
            tensor: wf
        """
        wf = self.data[index].astype('float32')
        mc = self.max_chans[index]
        if self.detected_spikes:
            y = [-1] # dummy value
        else:
            y = self.targets[index].astype('long')
        chan_loc = self.channel_locs[index].astype('float32')

        if self.transform is not None and self.use_chan_pos:
            wf, chan_loc = self.transform([wf, mc, chan_loc])
        elif self.transform is not None:
            wf = self.transform([wf, mc])
        
        if self.target_transform is not None:
            y = self.target_transform(y)

        if self.use_chan_pos:
            return [wf, chan_loc], y
        
        return wf, y


    def __len__(self) -> int:
        return len(self.data)


class WF_MultiChan_Dataset(Dataset):
    train_set_fn = "spikes_train.npy"
    spike_mcs_fn = "channel_num_train.npy"
    targets_fn = "labels_train.npy"
    chan_coords_fn = "channel_spike_locs_train.npy"

    def __init__(
        self,
        root: str,
        use_chan_pos: bool = False, 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        detected_spikes: bool = False
    ) -> None:

        super().__init__()

        self.data: Any = []

        # now load the numpy array
        self.data = np.load(os.path.join(root, self.train_set_fn))
        print(self.data.shape)
        self.root = root
        self.chan_nums = np.load(os.path.join(root, self.spike_mcs_fn))
        self.transform = transform
        self.detected_spikes = detected_spikes

        if detected_spikes:
            self.targets = None
            self.target_transform = None
        else:
            self.targets = np.load(os.path.join(root, self.targets_fn))
            self.target_transform = target_transform
        self.channel_locs = np.load(os.path.join(root, self.chan_coords_fn))
        self.use_chan_pos = use_chan_pos
        print(len(self.chan_nums))
        print(self.chan_nums.shape)

    def __getitem__(self, index: int) -> Any :
        """
        Args:
            index (int): Index
        Returns:
            tensor: wf
        """
        wf = self.data[index].astype('float32')
        if self.detected_spikes:
            y = [-1] # dummy value
        else:
            y = self.targets[index].astype('long')
        chan_nums = self.chan_nums[index]
        chan_loc = self.channel_locs[index].astype('float32')

        # doing this so that it is a tensor
        # wf = torch.from_numpy(wf)

        if self.transform is not None and self.use_chan_pos:
            wf, chan_loc = self.transform([wf, chan_nums, chan_loc])
        elif self.transform is not None:
            wf = self.transform([wf, chan_nums])

        if self.target_transform is not None:
            y = self.target_transform(y)
        
        if self.use_chan_pos:
            return [wf, chan_loc], y
            
        return wf, y


    def __len__(self) -> int:
        return len(self.data)


class WFDataset_lab(Dataset):
    train_set_fn = "spikes_train.npy"
    train_targets_fn = "labels_train.npy"
    spike_mcs_train_fn = "channel_num_train.npy"
    chan_coords_train_fn = "channel_spike_locs_train.npy"

    test_set_fn = "spikes_test.npy"
    test_targets_fn = "labels_test.npy"
    spike_mcs_test_fn = "channel_num_test.npy"
    chan_coords_test_fn = "channel_spike_locs_test.npy"

    val_set_fn = "spikes_val.npy"
    val_targets_fn = "labels_val.npy"
    spike_mcs_val_fn = "channel_num_val.npy"
    chan_coords_val_fn = "channel_spike_locs_val.npy"
    
    def __init__(
        self,
        root: str,
        use_chan_pos: bool = False, 
        multi_chan: bool = False,
        split: str = 'train',
        transform: Optional[Callable] = None,
        
    ) -> None:

        super().__init__()
        if split == 'train':
            print(multi_chan, self.train_set_fn)
            self.data = np.load(os.path.join(root, self.train_set_fn)).astype('float32')
            self.targets = np.load(os.path.join(root, self.train_targets_fn))
            self.chan_nums = np.load(os.path.join(root, self.spike_mcs_train_fn))
            self.channel_locs = np.load(os.path.join(root, self.chan_coords_train_fn))
        elif split == 'test':
            self.data = np.load(os.path.join(root, self.test_set_fn)).astype('float32')
            self.targets = np.load(os.path.join(root, self.test_targets_fn))
            self.chan_nums = np.load(os.path.join(root, self.spike_mcs_test_fn))
            self.channel_locs = np.load(os.path.join(root, self.chan_coords_test_fn))
        elif split == 'val':
            self.data = np.load(os.path.join(root, self.val_set_fn)).astype('float32')
            self.targets = np.load(os.path.join(root, self.val_targets_fn))
            self.chan_nums = np.load(os.path.join(root, self.spike_mcs_val_fn))
            self.channel_locs = np.load(os.path.join(root, self.chan_coords_val_fn))
            
        # self.data: Any = []

        # now load the numpy array
        print(self.data.shape)
        self.root = root
        self.transform = transform
        self.use_chan_pos = use_chan_pos
        self.num_classes = len(np.unique(self.targets))

    def __getitem__(self, index: int) -> Any :
        """
        Args:
            index (int): Index
        Returns:
            tensor: wf
        """
        wf = self.data[index].astype('float32')
        y = self.targets[index].astype('long')
        chan_nums = self.chan_nums[index]
        chan_loc = self.channel_locs[index].astype('float32')

        # doing this so that it is a tensor
        # wf = torch.from_numpy(wf)
        if self.transform is not None and self.use_chan_pos:
            wf, chan_loc = self.transform([wf, chan_nums, chan_loc])
        elif self.transform is not None:
            wf = self.transform([wf, chan_nums])

        if self.use_chan_pos:
            return [wf, chan_loc], y

        return wf, y

    def __len__(self) -> int:
        return len(self.data)

class ContrastiveLearningDataset:
    def __init__(self, root_folder, lat_dim, multi_chan, use_chan_pos=False):
        self.root_folder = root_folder
        self.lat_dim = lat_dim
        self.multi_chan = multi_chan
        self.use_chan_pos = use_chan_pos

    @staticmethod
    def get_wf_pipeline_transform(self, temp_cov_fn, spatial_cov_fn, noise_scale, num_extra_chans, normalize=False, p_crop=0.5):
        temporal_cov = np.load(os.path.join(self.root_folder, temp_cov_fn))
        spatial_cov = np.load(os.path.join(self.root_folder, spatial_cov_fn))
        """Return a set of data augmentation transformations on waveforms."""
        data_transforms = transforms.Compose([
                                            transforms.RandomApply([Collide(self.root_folder, multi_chan=self.multi_chan)], p=0.4),
                                            Crop(prob=p_crop, num_extra_chans=num_extra_chans),
                                            transforms.RandomApply([AmpJitter()], p=0.7),
                                              transforms.RandomApply([Jitter()], p=0.6),
                                            #   transforms.RandomApply([PCA_Reproj(root_folder=self.root_folder)], p=0.4),
                                            #   transforms.RandomApply([SmartNoise(self.root_folder, temporal_cov, 
                                            #                                      spatial_cov, noise_scale, normalize)], p=0.5),
                                              TorchToWfTensor()])
        
        return data_transforms

    @staticmethod
    def get_pca_transform(self):
        data_transforms = transforms.Compose([PCA_Reproj(root_folder=self.root_folder, pca_dim=self.lat_dim),
                                              ToWfTensor()])
        
        return data_transforms

    def get_dataset(self, name, n_views, noise_scale=1.0, num_extra_chans=0, normalize=False, p_crop=0.5, detected_spikes=False):
        temp_cov_fn = 'temporal_cov_example.npy'    
        spatial_cov_fn = 'spatial_cov_example.npy'
        if self.multi_chan:
            name = name + '_multichan'
        valid_datasets = {'wfs': lambda: WFDataset(self.root_folder,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_wf_pipeline_transform(self, temp_cov_fn,
                                                                  spatial_cov_fn,
                                                                #   noise_scale), self.get_pca_transform(self),
                                                                  noise_scale, 0, normalize), None,
                                                                  n_views),
                                                                  target_transform=LabelViewGenerator(),
                                                                  detected_spikes=detected_spikes),
                          'wfs_multichan': lambda: WF_MultiChan_Dataset(self.root_folder, use_chan_pos=self.use_chan_pos,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_wf_pipeline_transform(self, temp_cov_fn,
                                                                  spatial_cov_fn,
                                                                #   noise_scale), self.get_pca_transform(self),
                                                                  noise_scale, num_extra_chans, p_crop=p_crop), None,
                                                                  n_views),
                                                                  target_transform=LabelViewGenerator(),
                                                                  detected_spikes=detected_spikes)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise ValueError(f'Dataset {name} not supported or not existent')
        else:
            return dataset_fn()