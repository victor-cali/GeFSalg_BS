# Dataclasses Genotype and Gene
from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np

@dataclass(frozen=True)
class Gene:

    _selected_funcs: str
    _params: tuple
    _source: Union[tuple,str]
    _idx: int

    def __repr__(self) -> str:
        rep = {
            "selected_funcs": self._selected_funcs,
            "params": dict(self.params),
            "source": self.source,
            "idx": self.idx
        }
        key = 'freq_bands'
        if key in rep["params"]:
            rep["params"][key] = dict(rep["params"][key])
        return repr(rep)
    
    @property
    def selected_funcs(self):
        return self._selected_funcs

    @property
    def feature(self):
        return self._selected_funcs

    @property
    def params(self):
        args = dict(self._params)
        k1 = 'freq_bands'
        k2 = 'edge'
        if k1 in args:
            args[k1] = dict(args[k1])
        if k2 in args:
            args[k2] = list(args[k2])
        return args
    @property
    def source(self):
        return self._source
    @property
    def idx(self):
        return self._idx

@dataclass()
class Genotype():

    genes: Tuple[Gene]
    score: float = -1.0

    def __len__(self) -> int:
        return len(self.genes)

    def __repr__(self) -> str:
        rep = [g.__dict__ for g in self.genes]
        return repr({'genes':rep, 'score': self.score})

    def __iter__(self):
        self._g = 0
        return self

    def __next__(self):
        if self._g < self.__len__():
            self._g += 1
            return self.genes[self._g-1]
        else:
            raise StopIteration
    
    def __lt__(self, obj) -> bool:
        if type(obj) is not Genotype:
            raise TypeError
        return (self.score < obj.score)
    
    def __gt__(self, obj) -> bool:
        if type(obj) is not Genotype:
            raise TypeError
        return (self.score > obj.score)
    
    def __le__(self, obj) -> bool:
        if type(obj) is not Genotype:
            raise TypeError
        return (self.score <= obj.score)
    
    def __ge__(self, obj) -> bool:
        if type(obj) is not Genotype:
            raise TypeError
        return (self.score >= obj.score)
    
    def __getitem__(self, i):
        return self.genes[i]
