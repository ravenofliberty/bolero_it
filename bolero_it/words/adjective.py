import logging

from bolero_it.words.word import Word

logger = logging.getLogger(__name__)

"""
Word object handles all states. Persistence manager (PM) only pulls and pushes updates to the DB. 
Word object only interacts with the DB via PM. 
"""


class Adjective(Word):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return f"{self.word.lower()} ({str(self.cls.name)}) - " \
               f"Eng: {self.meaning}"


