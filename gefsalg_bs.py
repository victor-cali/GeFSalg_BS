import csv
import warnings
import itertools
import numpy as np
from sklearn import svm
from mne.epochs import BaseEpochs
from GeFSalg_BS.utils import Utils
from GeFSalg_BS.dna import Gene, Genotype
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from GeFSalg_BS.solutionspace import SolutionSpace
from mne_features.feature_extraction import extract_features
from scipy.stats import spearmanr, SpearmanRConstantInputWarning

warnings.filterwarnings("error")
np.seterr(divide='ignore', invalid='ignore')

class GenAlgo():

    best: Genotype
    niche: list
    genoma: tuple
    parents: list
    extintion = 0
    generation = 0
    offspring: list
    population: list
    genotype: Genotype
    progress_counter = 0
    phenotype: np.ndarray
    mutate_edit_counter = 0
    rng = np.random.default_rng()
    mutation_rates = [0.1,0.2,0.7]

    #  Crear poblacià¸£à¸“n -> Mapear -> Evaluar -> Elegir padres -> Cruzar -> Mutar -> Ajustes Multimodales
    def __init__(self,
        out_path: str, 
        epochs: BaseEpochs,
        genotype_len: int = 4,
        extintions_lim: int = 100,
        population_len: int = 100,
        survival_rate: float = 0.1,
        generations_lim: int = 1000
    ) -> None:
        # Input arguments
        self.out_path = out_path
        self.epochs = epochs
        self.genotype_len = genotype_len
        self.extintions_lim = extintions_lim
        self.population_len = population_len
        self.survival_rate = survival_rate
        self.generations_lim = generations_lim
        # Other parameters for the evolution cycle and data obtention
        self.parents_len = int(self.population_len*self.survival_rate)
        self.niche_len = int(self.population_len*self.survival_rate)
        self.tournament_len = self.parents_len // 2
        self.genoma_len = self.genotype_len * self.parents_len
        self.offspring_len = self.population_len - self.parents_len - self.niche_len
        # Support objects
        self._cache = dict()
        self.utils = Utils(epochs, genotype_len)
        self.solution_space = SolutionSpace(self.utils)
        self.folds_number = 2
        self.fitness_func = svm.SVC(kernel='rbf')
        self.skf = StratifiedKFold(n_splits=self.folds_number)
        self.extintion_fate = self.rng.choice(np.arange(30,100,10))
        #
        self.phenotype = np.zeros(self.utils.pheno_shape)

        header = []
        for i in range(self.genotype_len):
            header.append(f'Gene{i+1}') 
        header += ['Generation','Extintions','Fitness']
        with open(self.out_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def __call__(self, *args: any, **kwds: any) -> any:
        self.population = self.solution_space.build_population(self.population_len)
        
        self.best = max(self.population)

        while self.generation < self.generations_lim:#self.generations_lim
            
            self.map_population()

            self.rate_population()
            
            self.population.sort()
            self.population = self.population[self.population_len//2:]
            
            self.select_parents()

            self.make_cross_over()

            self.evolve_niche()

            self.mutate_offspring()

            self.update_mutation_rates()

            self.set_next_generation()

            self.record_generation()

            print(f'Generation: {self.generation}, Extintions: {self.extintion}, Best: {self.best.score}')
        print(f'FINISHED\n Best Candidate: {self.best.score}')
        return self.best
    
    def map_population(self) -> None:
        genome = [gene for genotype in self.population for gene in genotype]
        for key in set(self._cache) - set(genome):
            del self._cache[key]
        for gene in genome:
            # Get every gene
            if gene not in self._cache.keys():
                # Get function name
                func = gene.selected_funcs
                data = self.utils.filterbank.get_data(picks = gene.source)
                args = gene.params
                params = {f'{func}__{key}': val for key,val in args.items()}
                try:
                    feature = extract_features(
                        X = data, sfreq = self.utils.fs, 
                        selected_funcs = [func], funcs_params = params
                    )
                    # SOLVE FEATURE FORM
                    # When no rate is obtained (1 source)
                    # no selection applied
                    if feature.shape[-1] == 1:
                        phenotype = feature[:,0]
                    # When rate is obtained from 2 sources 
                    # no selection applied
                    elif gene.idx == -1 and feature.shape[-1] == 2:
                        phenotype = feature[:,0]/feature[:,1]
                    # When rate is obtained from 2 sources 
                    # selection applied
                    elif feature.shape[-1] == 8 and gene.selected_funcs == 'spect_slope':
                        j = gene.idx
                        phenotype = feature[:,j]/feature[:,j+4]
                    # When no rate is obtained (1 source)
                    # selection applied
                    else:
                        phenotype = feature[:,gene.idx]
                    # Clean from nan, inf and -inf vals
                    if np.any(np.isnan(phenotype)) or np.any(np.isinf(phenotype)):
                        try:
                            np.nan_to_num(phenotype, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                val = np.nanmean(phenotype)
                            np.nan_to_num(phenotype, copy=False, nan=val, posinf=val, neginf=val)
                        except Exception as e:
                            pass
                    phenotype = StandardScaler().fit_transform(phenotype.reshape(-1, 1))
                except Exception as e:
                    phenotype = np.zeros((len(self.epochs),1))
                self._cache.update({gene: phenotype[:,0]})

    def rate_population(self) -> None:
        def make_X() -> np.ndarray:
            x = np.zeros(self.utils.pheno_shape)
            for g in range(self.genotype_len):
                gene = genotype[g]
                x[:, g] = self._cache.get(gene)
            return x
        for genotype in self.population:
            X = make_X()
            y = self.epochs.events[:, -1]
            fitness = 0
            for train_index, test_index in self.skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                #Train the model using the training sets
                try:
                    self.fitness_func.fit(X_train, y_train)
                    accuracy = self.fitness_func.score(X_test, y_test)
                except:
                    accuracy = 0.5
                fitness += accuracy
            fitness /= self.folds_number
            score = 200.0*fitness-100.0
            genotype.score = score

    def select_parents(self):
        guide = set()
        self.parents = list()
        best = max(self.population)
        guide.add(best.genes)
        self.parents.append(best)
        while len(guide) < self.parents_len:
            selected = self.rng.choice(
                np.arange(self.population_len//2), 
                self.tournament_len, replace=False
            )
            competitors = [self.population[i] for i in selected]
            winner = max(competitors)
            if winner.genes not in guide:
                guide.add(winner.genes)
                self.parents.append(winner)
        self.genoma = tuple(itertools.chain(*guide))

    def make_cross_over(self) -> None:
        genome = [gene for genotype in self.population for gene in genotype]
        reference_genes = self.rng.choice(len(self.genoma), size = self.offspring_len//2, replace = False)
        offspring = [[self.genoma[g]] for g in reference_genes]
        for new_genotype in offspring:
            for _ in range(self.genotype_len-1):
                correlations = []
                for gene in genome:
                    correlations.append(self.check_corr(new_genotype, gene))
                smallest_correlation = min(correlations)
                g = correlations.index(smallest_correlation)
                least_correlated_gene = genome[g]
                new_genotype.append(least_correlated_gene)
        self.offspring = [gene for genotype in offspring for gene in genotype]

        guide = set()
        while len(guide) < self.offspring_len//2:
            selected = self.rng.choice(
                np.arange(self.genoma_len),
                self.genotype_len, replace = False
            )
            genes = tuple([self.genoma[i] for i in selected])
            if genes not in guide:
                guide.add(genes)
                self.offspring += genes

    def check_corr(self, new_genotype, new_gene):
        correlation = 0
        for gene in new_genotype:
            phenotype1 = self._cache.get(gene)
            phenotype2 = self._cache.get(new_gene)
            try:
                corr, _ = spearmanr(phenotype1, phenotype2)
            except SpearmanRConstantInputWarning:
                corr = 1
            correlation += abs(corr)
        correlation /= len(new_genotype)
        return correlation

    def map_geno_to_pheno(self, genotype = None):
        if genotype is not None:
            assert type(genotype) is Genotype
            for gene in genotype:
                assert type(gene) is Gene
            self.genotype = genotype

        self.phenotype = np.zeros(self.utils.pheno_shape)
        for i in range(self.genotype_len): 
            # Get every gene
            if self.genotype[i] in self._cache.keys():
                self.phenotype[:,i] = self._cache.get(self.genotype[i])
            else:
                # Get function name
                func = self.genotype[i].selected_funcs
                data = self.utils.filterbank.get_data(picks = self.genotype[i].source)
                args = self.genotype[i].params
                params = {f'{func}__{key}': val for key,val in args.items()}
                try:
                    feature = extract_features(
                    X = data, sfreq = self.utils.fs, 
                    selected_funcs = [func], funcs_params = params)
                    # SOLVE FEATURE FORM
                    # When no rate is obtained (1 source)
                    # no selection applied
                    if feature.shape[-1] == 1:
                        self.phenotype[:,i] = feature[:,0]
                    # When rate is obtained from 2 sources 
                    # no selection applied
                    elif self.genotype[i].idx == -1 and feature.shape[-1] == 2:
                        self.phenotype[:,i] = feature[:,0]/feature[:,1]
                    # When rate is obtained from 4 sources 
                    # no selection applied
                    elif self.genotype[i].idx == -1 and feature.shape[-1] == 4:
                        temp1 = feature[:,0]/feature[:,1]
                        temp2 = feature[:,2]/feature[:,3]
                        self.phenotype[:,i] = temp1/temp2
                    # When rate is obtained from 2 sources 
                    # selection applied
                    elif feature.shape[-1] == 8 and self.genotype[i].selected_funcs == 'spect_slope':
                        j = self.genotype[i].idx
                        self.phenotype[:,i] = feature[:,j]/feature[:,j+4]
                    # When rate is obtained from 4 sources 
                    # selection applied
                    elif feature.shape[-1] == 16 and self.genotype[i].selected_funcs == 'spect_slope':
                        j = self.genotype[i].idx
                        self.phenotype[:,i] = feature[:,j]/feature[:,j+4]
                        temp1 = feature[:,j]/feature[:,j+4]
                        temp2 = feature[:,j+8]/feature[:,j+12]
                        self.phenotype[:,i] = temp1/temp2
                    # When no rate is obtained (1 source)
                    # selection applied
                    else:
                        self.phenotype[:,i] = feature[:,self.genotype[i].idx]
                except Exception as e: 
                    self.phenotype[:,i] = np.zeros(len(self.epochs))
                self._cache.update({self.genotype[i]: self.phenotype[:,i]})
        # Clean from nan, inf and -inf vals
        if np.any(np.isnan(self.phenotype)) or np.any(np.isinf(self.phenotype)):
            try:
                np.nan_to_num(self.phenotype,copy=False,nan=np.nan,posinf=np.nan,neginf=np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    val = np.nanmean(self.phenotype,axis=0)
                np.nan_to_num(self.phenotype,copy=False,nan=val,posinf=val,neginf=val)
            except:
                pass
            if np.any(np.isnan(self.phenotype)) or np.any(np.isinf(self.phenotype)):
                np.nan_to_num(self.phenotype,copy=False)
        self.phenotype = StandardScaler().fit_transform(self.phenotype)
        if genotype is not None:
            return self.phenotype

    def calculate_score(self, phenotype = None):
        if phenotype is not None:
            self.phenotype = phenotype
        X = self.phenotype
        y = self.epochs.events[:, -1]
        fitness = 0
        for train_index, test_index in self.skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #Train the model using the training sets
            try:
                self.fitness_func.fit(X_train, y_train)
                accuracy = self.fitness_func.score(X_test, y_test)
            except:
                accuracy = 50.0
            fitness += accuracy
        fitness /= self.folds_number
        score = 200.0*fitness-100.0
        self.genotype.score = score
        if phenotype is not None:
            return score

    def cross_over(self):
        guide = set()
        self.offspring = list()
        while len(guide) < self.offspring_len:
            selected = self.rng.choice(
                np.arange(self.genoma_len),
                self.genotype_len, replace = False
            )
            genes = tuple([self.genoma[i] for i in selected])
            if genes not in guide:
                guide.add(genes)
                self.offspring += genes

    def mutate_offspring(self):
        for i in range((self.genotype_len * self.offspring_len)):
            toss = self.rng.choice(3, p=self.mutation_rates)
            functional_gene = np.count_nonzero(self._cache.get(self.offspring[i]))
            if not functional_gene or toss == 0:
                mutation = self.solution_space.build_gene()
                del self.offspring[i]
                self.offspring.insert(i, mutation)
            elif toss == 1:
                mutation = self.solution_space.reset_gene(self.offspring[i])
                del self.offspring[i]
                self.offspring.insert(i, mutation)
            else:
                pass
        offspring = list()
        for _ in range(self.offspring_len):
            genes = [self.offspring.pop(0) for __ in range(self.genotype_len)]
            offspring.append(Genotype(tuple(genes)))
        self.offspring = offspring

    def evolve_niche(self):
        self.niche = list()
        half_niche_len = self.niche_len//2
        for _ in range(half_niche_len):
            selected = self.rng.integers(self.parents_len)
            genotype = self.parents[selected]
            mutation = [self.solution_space.reset_gene(gene) for gene in genotype.genes]
            self.niche.append(Genotype(tuple(mutation)))

        guide = [best.genes for best in self.population[-half_niche_len:]]
        genoma = tuple(itertools.chain(*guide))
        guide = self.rng.permutation(half_niche_len*self.genotype_len)
        genoma = [genoma[g] for g in guide]
        for _ in range(half_niche_len):
            genes = [genoma.pop(0) for __ in range(self.genotype_len)]
            self.niche.append(Genotype(tuple(genes)))

    def set_next_generation(self):
        if self.progress_counter <= self.extintion_fate:
            self.population = self.offspring + self.parents + self.niche
            self.generation += 1
        elif self.extintion < self.extintions_lim:
                self.population = self.solution_space.build_population(self.population_len - 1 )
                self.extintion_fate = self.rng.choice(np.arange(50,100,10))
                self.population.append(self.best)
                self.progress_counter = 0
                self.extintion += 1
                self.generation = 0
        else:
            self.population = self.offspring + self.parents + self.niche
            self.generation += 1
            
    def record_generation(self):
        with open(self.out_path, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for genotype in self.parents:
                row = [repr(gene) for gene in genotype]
                row += [self.generation-1, self.extintion, genotype.score]
                writer.writerow(row)

    def update_mutation_rates(self):

        new_best = max(self.parents)
        progress = new_best.score - self.best.score
        
        if progress < 0.1:
            self.progress_counter += 1
            if self.mutation_rates[1]<0.35:
                self.mutation_rates[1] += 0.050
                self.mutation_rates[2] -= 0.050
            elif self.mutation_rates[2] > 0.3:
                self.mutation_rates[0] += 0.050
                self.mutation_rates[2] -= 0.050
            else:
                if self.mutate_edit_counter>=3:
                    self.mutation_rates = [0.1,0.2,0.7] 
                    self.mutate_edit_counter = 0
                else: 
                    self.mutate_edit_counter +=1 
        elif progress > 0.1:
            self.progress_counter -= 1 if progress < 1 else int(progress)
            if self.progress_counter < 0: self.progress_counter = 0

            if self.mutation_rates[2]<0.5:
                self.mutation_rates[0] -= 0.050
                self.mutation_rates[2] += 0.050
            elif self.mutation_rates[2]<0.70:
                self.mutation_rates[1] -= 0.050
                self.mutation_rates[2] += 0.050
            else:
                if self.mutate_edit_counter>=3:
                    self.mutation_rates = [0.1,0.2,0.7] 
                    self.mutate_edit_counter = 0
                else: 
                    self.mutate_edit_counter +=1
        self.best = new_best