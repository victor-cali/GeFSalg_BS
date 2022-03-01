# Parameters Class
import numpy as np
from sklearn.neighbors import KDTree
from mne_features.bivariate import get_bivariate_funcs
from mne_features.univariate import get_univariate_funcs

class Utils:

    fbands = ((0.5,4),(4,7.5),(7.5,13),(7.5,17.5),(7.5,22),(7.5,30),(13,17.5),(13,30),(17.5,24),(17.5,30))
    
    def __init__(self, epochs, genotype_len):

        self.pheno_shape = (len(epochs),genotype_len)
        self.genotype_len = genotype_len
        self.filterbank = epochs.copy()
        self.fs = epochs.info['sfreq']
        self.epochs = epochs.copy()
        self.params = dict()
        self._set_funcs()

        # Embedding dimension
        self.params["emb"] = np.arange(2,21)
        # Name of the metric function used with KDTree
        self.params["metric"] = np.array(KDTree.valid_metrics,dtype=object)
        # Freq bands, Fmin and Fmax
        self.params["freq_bands"] = {i: band for i, band in zip(range(len(self.fbands)), self.fbands)}
        # Method used for the estimation 
        # of the Power Spectral Density PSD
        self.params["psd_method"] = np.array(['welch','multitaper','fft'],dtype=object)
        # Maximum delay/offset (in number of samples)
        self.params["kmax"] = np.arange(5,18)
        # Delay (number of samples)
        self.params["tau"] = np.arange(1,6)
        # If a derivative filter is applied or not to the input data
        self.params["deriv_filt"] = np.array([True,False],dtype=bool)
        # Reference frequency for the computation of the spectral edge frequency
        self.params["ref_freq"] = np.array([self.fs//frac for frac in range(2,6)])
        # Expected to be a list of values between 0 and 1
        self.params["edge"] = np.arange(0,1,0.1)
        # Quantile or sequence of quantiles to compute 
        # which must be between 0 and 1 inclusive.
        self.params["q"] = np.arange(0,1,0.1)
        # Number of Nearest Neighbors
        self.params["nn"] = np.arange(2,10)

        self._set_filterbank()

        # Non-filtered channels names
        self.params["original_channels"] = self.epochs.ch_names
        # Filtered channels names
        self.params["filtered_channels"] = self.filterbank.ch_names

    def _set_filterbank(self):
        new_names = dict()
        filtered_epochs = list()
        for band in self.fbands:
            new_names.clear()
            for name in self.epochs.ch_names:
                new_names.update({name:f'{name}({band[0]}, {band[1]})'})
            subepochs = self.epochs.copy()
            subepochs.filter(band[0], band[1], method = 'iir', verbose = 50)
            subepochs.rename_channels(new_names)
            filtered_epochs.append(subepochs.copy())
        self.filterbank.add_channels(filtered_epochs,force_update_info=True)

    def _set_funcs(self):
        univariate_funcs = get_univariate_funcs(self.fs)
        bivariate_funcs = get_bivariate_funcs(self.fs)
        self.funcs = {**univariate_funcs, **bivariate_funcs}
        not_used_funcs = ('wavelet_coef_energy', 'teager_kaiser_energy')
        for not_used_func in not_used_funcs: del self.funcs[not_used_func]
        self.params['funcs'] = np.array(list(self.funcs.keys()), dtype = object)