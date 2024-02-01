# Slightly modified BracketParseCorpusReader to handle treebank_3 prd files and combined_treebank parse files.

import sys

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.corpus import BracketParseCorpusReader

# we use [^\s()]+ instead of \S+? to avoid matching ()
SORTTAGWRD = re.compile(r"\((\d+) ([^\s()]+) ([^\s()]+)\)")
TAGWORD = re.compile(r"\(([^\s()]+) ([^\s()]+)\)")
WORD = re.compile(r"\([^\s()]+ ([^\s()]+)\)")
EMPTY_BRACKETS = re.compile(r"\s*\(\s*\(")


class CustomizedBracketParseCorpusReader(BracketParseCorpusReader):
    def __init__(self, *args, **kwargs):
        BracketParseCorpusReader.__init__(self, *args, **kwargs)

    def _normalize(self, t):
        # Replace leaves of the form (!), (,), with (! !), (, ,)
        t = re.sub(r"\((.)\)", r"(\1 \1)", t)
        # Seems we don't have this format in treebank_3 prd files, so comment it out
        # Otherwise, we end up with over kill on text tokens
        # Replace leaves of the form (tag word root) with (tag word)
        # t = re.sub(r"\(([^\s()]+) ([^\s()]+) [^\s()]+\)", r"(\1 \2)", t)
        return t