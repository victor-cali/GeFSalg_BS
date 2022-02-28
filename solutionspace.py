# Solution Space Class
from mne_features.univariate import get_univariate_funcs
from mne_features.bivariate import get_bivariate_funcs
from GeFSalg_BS.dna import Gene, Genotype
from GeFSalg_BS.utils import Utils
from inspect import getfullargspec
import numpy as np

class SolutionSpace:

    _rng = np.random.default_rng()

    def __init__(self, utils: Utils) -> None:
        self._gene = dict()
        self._genotype = None
        self._params = utils.params
        self.genotype_len = range(utils.genotype_len)
        self._un_funcs = get_univariate_funcs(utils.epochs.info['sfreq'])
        self._bi_funcs = get_bivariate_funcs(utils.epochs.info['sfreq'])

    def build_population(self, size: int) -> list:
        new_population = list()
        for _ in range(size):
            self.build_genotype()
            new_population.append(self._genotype)
        
        return new_population

    def build_genotype(self) -> Genotype:
        genes = []
        for _ in self.genotype_len:
            self.build_gene()
            genes.append(self._gene)
        self._genotype = Genotype(tuple(genes))
        return self._genotype

    def build_gene(self) -> Gene:
        self._gene = dict()
        # Parameters for the function
        self._gene["params"] = dict()
        # Index for special functions
        self._gene["index"] = -1
        # Channels to be used
        self._gene["source"] = None

        self.choose_feature()

        # Selected Function 
        func = self._gene["selected_funcs"]
        # Specific parameters obtention
        if func in self._un_funcs.keys():
            self.args = set(getfullargspec(self._un_funcs[func])[0])
        elif func in self._bi_funcs.keys():
            self.args = set(getfullargspec(self._bi_funcs[func])[0])
        else:
            raise Exception    
        self.args.discard('data')

        #self._set_gene()
        self.choose_source()
        self.choose_params()

        self._gene = Gene(
            self._gene['selected_funcs'],
            tuple(self._gene['params'].items()),
            self._gene['source'],
            self._gene['index']
        )
        return self._gene

    def reset_gene(self,gene):
        self._gene = dict()
        # Parameters for the function
        self._gene["params"] = dict()
        # Index for special functions
        self._gene["index"] = -1
        # Channels to be used
        self._gene["source"] = None

        self._gene['selected_funcs'] = gene.feature

        # Selected Function 
        func = self._gene["selected_funcs"]
        # Specific parameters obtention
        if func in self._un_funcs:
            self.args = set(getfullargspec(self._un_funcs[func])[0])
        elif func in self._bi_funcs:
            self.args = set(getfullargspec(self._bi_funcs[func])[0])
        else:
            raise Exception    
        self.args.discard('data')

        #self._set_gene()
        self.choose_source()
        self.choose_params()

        self._gene = Gene(
            self._gene['selected_funcs'],
            tuple(self._gene['params'].items()),
            self._gene['source'],
            self._gene['index']
        )
        return self._gene

    def choose_feature(self) -> None:
        self._gene['selected_funcs'] = self._rng.choice(self._params['funcs'])

    def choose_source(self) -> None:
        func = self._gene["selected_funcs"]
        # 1 canal y 2 bandas 
        # 2 canales y 1 banda 
        # 1 canal 1 banda
        if 'freq_bands' in self.args:
            key = 'freq_bands'
            case = self._rng.integers(1,3)
            selection = self._rng.choice(len(self._params[key]), size = case, replace = False)
            value = {f'band {i}': self._params[key][i] for i in selection}
            self._gene["params"].update({key: tuple(value.items())})
            self.args.discard(key)
            # Special channel selection for univariate functions 
            # with self contained frequency settings
            key = 'original_channels'
            if case == 1:
                case = self._rng.integers(1,3)
            else:
                case = 1
            selection = self._rng.choice(self._params[key], size = case, replace = False)
            self._gene["source"] = tuple(selection)
        elif {'fmin','fmax'}.issubset(self.args):
            key = 'freq_bands'
            key1 = 'fmin' 
            key2 = 'fmax' 
            selection = self._rng.choice(len(self._params[key]))
            fmin, fmax = self._params[key][selection]
            self._gene["params"].update({key1: fmin})
            self._gene["params"].update({key2: fmax})
            self.args.discard(key1)
            self.args.discard(key2)
            # Special channel selection for univariate functions 
            # with self contained frequency settings
            key = 'original_channels'
            case = self._rng.integers(1,3)
            selection = self._rng.choice(self._params[key], size = case, replace = False)
            self._gene["source"] = tuple(selection)
        # Channel selection for bivariate functions
        # with no self contained frequency settings
        elif func in self._bi_funcs.keys():
            key = 'original_channels'
            ch_selection = self._rng.choice(self._params[key], size = 2, replace = False)
            key = 'freq_bands'
            index = self._rng.choice(len(self._params[key]))
            fb_selection = self._params[key][index]
            src = [f'{ch}{fb_selection}' for ch in ch_selection]
            self._gene["source"] = tuple(src)
        # Channel selection for univariate functions 
        # with no self contained frequency settings
        else:
            key = 'original_channels'
            case = self._rng.integers(1,3)
            ch_selection = self._rng.choice(self._params[key], size = case, replace = False)
            key = 'freq_bands'
            if case == 1:
                case = self._rng.integers(1,3)
            else:
                case = 1
            indexes = self._rng.choice(len(self._params[key]), size = case, replace = False)
            fb_selection = [self._params[key][i] for i in indexes]

            if len(ch_selection) == 1 and len(fb_selection) == 1:
                self._gene["source"] = f'{ch_selection[0]}{fb_selection[0]}'
            elif len(ch_selection) == 1 and len(fb_selection) == 2:
                src = [f'{ch_selection[0]}{fb}' for fb in fb_selection]
                self._gene["source"] = tuple(src)
            elif len(ch_selection) == 2 and len(fb_selection) == 1:
                src = [f'{ch}{fb_selection[0]}' for ch in ch_selection]
                self._gene["source"] = tuple(src)

    def choose_params(self) -> None:
        # Set default parameters
        # All parameters that are predefined
        # and should stay always equal are default
        # Default settings, not chosen randomly
        if 'normalize' in self.args:
            key = 'normalize'
            self._gene["params"].update({key: False})
            self.args.discard(key)
        if 'ratios' in self.args:
            key = 'ratios'
            self._gene["params"].update({key: None})
            self.args.discard(key)
        if 'include_diag' in self.args:
            key = 'include_diag'
            self._gene["params"].update({key: False})
            self.args.discard(key)
        if 'with_intercept' in self.args:
            key = 'with_intercept'
            self._gene["params"].update({key: True})
            self.args.discard(key)
            # Index selection for special function
            i = self._rng.integers(4)
            self._gene["index"] = int(i)
        if 'with_eigenvalues' in self.args:
            key = 'with_eigenvalues'
            self._gene["params"].update({key: False})
            self.args.discard(key)
            # Index selection for special function
            i = self._rng.integers(3)
            self._gene["index"] = int(i)
        
        # Set simple parameters
        # All parameters which setting does 
        # not need calculations are simple
        for arg in self.args:
            try:
                val = self._rng.choice(self._params[arg])
                self._gene["params"].update({arg: val})
            except KeyError:
                pass

    # Set all Gene object attributes
    def _set_gene(self) -> None:

        # Selected Function 
        func = self._gene["selected_funcs"]
        # Parameters for the function
        self._gene["params"] = dict()
        # Channels to be used
        self._gene["source"] = None
        # Index for special functions
        self._gene["index"] = -1

        # Specific parameters obtention
        if func in self._un_funcs.keys():
            args = set(getfullargspec(self._un_funcs[func])[0])
        elif func in self._bi_funcs.keys():
            args = set(getfullargspec(self._bi_funcs[func])[0])
        else:
            raise Exception
        args.discard('data')
        
        # Set default parameters
        # All parameters that are predefined
        # and should stay always equal are default
        # Default settings, not chosen randomly
        if 'normalize' in args:
            key = 'normalize'
            self._gene["params"].update({key: False})
            args.discard(key)
        if 'ratios' in args:
            key = 'ratios'
            self._gene["params"].update({key: None})
            args.discard(key)
        if 'include_diag' in args:
            key = 'include_diag'
            self._gene["params"].update({key: False})
            args.discard(key)
        if 'with_intercept' in args:
            key = 'with_intercept'
            self._gene["params"].update({key: True})
            args.discard(key)
            # Index selection for special function
            i = self._rng.integers(4)
            self._gene["index"] = int(i)
        if 'with_eigenvalues' in args:
            key = 'with_eigenvalues'
            self._gene["params"].update({key: False})
            args.discard(key)
            # Index selection for special function
            i = self._rng.integers(3)
            self._gene["index"] = int(i)

        # Set complex parameters
        # All parameters which setting 
        # involves calculations are complex
        if 'q' in args:
            key = 'q'
            case = self._rng.integers(1,3)
            value = self._rng.choice(self._params[key], size = case, replace = False)
            self._gene["params"].update({key: tuple(value)})
            args.discard(key)
        if 'edge' in args:
            key = 'edge'
            case = self._rng.integers(1,3)
            value = self._rng.choice(self._params[key], size = case, replace = False)
            self._gene["params"].update({key: tuple(value)})
            args.discard(key)
        # Frequency settings
        if 'freq_bands' in args:
            key = 'freq_bands'
            case = self._rng.integers(1,3)
            selection = self._rng.choice(len(self._params[key]), size = case, replace = False)
            value = {f'band {i}': self._params[key][i] for i in selection}
            self._gene["params"].update({key: tuple(value.items())})
            args.discard(key)
            # Special channel selection for univariate functions 
            # with self contained frequency settings
            key = 'original_channels'
            case = self._rng.integers(1,3)
            selection = self._rng.choice(self._params[key], size = case, replace = False)
            self._gene["source"] = tuple(selection)
        elif {'fmin','fmax'}.issubset(args):
            key = 'freq_bands'
            key1 = 'fmin' 
            key2 = 'fmax' 
            selection = self._rng.choice(len(self._params[key]))
            fmin, fmax = self._params[key][selection]
            self._gene["params"].update({key1: fmin})
            self._gene["params"].update({key2: fmax})
            args.discard(key1)
            args.discard(key2)
            # Special channel selection for univariate functions 
            # with self contained frequency settings
            key = 'original_channels'
            case = self._rng.integers(1,3)
            selection = self._rng.choice(self._params[key], size = case, replace = False)
            self._gene["source"] = tuple(selection)
        # Channel selection for bivariate functions
        # with no self contained frequency settings
        elif func in self._bi_funcs.keys():
            key = 'original_channels'
            ch_selection = self._rng.choice(self._params[key], size = 2, replace = False)
            key = 'freq_bands'
            indexes = self._rng.choice(len(self._params[key]), size = 2)

            fb_selection = [self._params[key][i] for i in indexes]
            src = [f'{ch}{fb}' for ch, fb in zip(ch_selection, fb_selection)]
            self._gene["source"] = tuple(src)
        # Channel selection for univariate functions 
        # with no self contained frequency settings
        else:
            key = 'original_channels'
            case = self._rng.integers(1,3)
            ch_selection = self._rng.choice(self._params[key], size = case, replace = False)
            key = 'freq_bands'
            case = self._rng.integers(1,3)
            indexes = self._rng.choice(len(self._params[key]), size = case, replace = False)
            fb_selection = [self._params[key][i] for i in indexes]

            if len(ch_selection) == 1 and len(fb_selection) == 1:
                self._gene["source"] = f'{ch_selection[0]}{fb_selection[0]}'
            elif len(ch_selection) == 1 and len(fb_selection) == 2:
                src = [f'{ch_selection[0]}{fb}' for fb in fb_selection]
                self._gene["source"] = tuple(src)
            elif len(ch_selection) == 2 and len(fb_selection) == 1:
                src = [f'{ch}{fb_selection[0]}' for ch in ch_selection]
                self._gene["source"] = tuple(src)
            elif len(ch_selection) == 2 and len(fb_selection) == 2:
                src1 = [f'{ch_selection[0]}{fb}' for fb in fb_selection]
                src2 = [f'{ch_selection[1]}{fb}' for fb in fb_selection]
                src = src1 + src2
                self._gene["source"] = tuple(src)

        # Set simple parameters
        # All parameters which setting does 
        # not need calculations are simple
        for arg in args:
            try:
                val = self._rng.choice(self._params[arg])
                self._gene["params"].update({arg: val})
            except KeyError:
                pass