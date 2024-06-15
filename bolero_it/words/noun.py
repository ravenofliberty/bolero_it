import logging

from bolero_it.words.word_metadata import ARTICLE_MAPPING_NOMINATIVE
from bolero_it.words.word import Word

logger = logging.getLogger(__name__)

"""
Word object handles all states. Persistence manager (PM) only pulls and pushes updates to the DB. 
Word object only interacts with the DB via PM. 
"""


class Noun(Word):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return f"" \
               f"{ARTICLE_MAPPING_NOMINATIVE[self.gender]['definite'].name} " \
               f"{self.word.capitalize()} ({str(self.cls.name)}) - " \
               f"Eng: {self.meaning}"
