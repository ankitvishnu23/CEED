import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Create two transformed views of a waveform and return."""

    def __init__(self, base_transform, freq_transform, n_views=2):
        self.base_transform = base_transform
        self.freq_transform = freq_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


class LabelViewGenerator(object):
    """Duplicate labels to use as pseudo-labels for augmented pairs of wfs."""

    def __init__(self, n_views=2):
        self.n_views = n_views

    def __call__(self, x):
        return [x, x]
