from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Words(Enum):
    Noun = 1
    Verb = 2
    Adjective = 3


class Tags(Enum):
    """
    Simple tags to help with generating random words that have something in common.
    """
    NA = 0
    Character = 1
    Color = 2
    Family = 3
    Politics = 4
    Economy = 5
    Taste = 6


class DefiniteArticle(Enum):
    il = 1
    la = 2
    lo = 3
    i = 4
    gli = 5
    le = 6


class IndefiniteArticle(Enum):
    un = 1
    una = 2
    uno = 3
    un_a = 4  # un'  - un - appostroph


class Gender(Enum):
    maschile = 1
    femminile = 2
    NA = 3


ARTICLE_MAPPING_NOMINATIVE = {
    Gender.maschile: {'definite': DefiniteArticle.il, 'indefinite': IndefiniteArticle.un},
    Gender.femminile: {'definite': DefiniteArticle.la, 'indefinite': IndefiniteArticle.una},
}


@dataclass
class VerbForms:
    io: str
    tu: str
    lui: str
    noi: str
    voi: str
    loro: str

    @classmethod
    def get_persons(cls):
        return ["io", "tu", "lui", "noi", "voi", "loro"]

    def to_json(self):
        return {
            "io": self.io,
            "tu": self.tu,
            "lui": self.lui,
            "noi": self.noi,
            "voi": self.voi,
            "loro": self.loro
        }

    @classmethod
    def from_json(cls, json):
        for k in json.keys():
            assert k in cls.get_persons(), f"Unknown person: {k}"
        return VerbForms(**json)

    # def __repr__(self):
    #     return self.to_json()
