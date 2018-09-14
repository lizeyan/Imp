import re
import unicodedata
from io import open

from snippets.data import ParallelDataset, SequenceDataset, Lang
from snippets.utilities import split


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        source_lang = Lang(lang2)
        target_lang = Lang(lang1)
    else:
        source_lang = Lang(lang1)
        target_lang = Lang(lang2)
    return source_lang, target_lang, pairs


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False, split_radio=None):
    if split_radio is None:
        split_radio = [0.5, 0.2, 0.3]
    source_lang, target_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    print(f"translate from {source_lang.name} to {target_lang.name}")
    source_lang.add_sentences([p[0] for p in pairs])
    target_lang.add_sentences([p[1] for p in pairs])
    train_pairs, valid_pairs, test_pairs = split(pairs, split_radio)
    train_dataset = ParallelDataset(
        SequenceDataset([p[0] for p in train_pairs], source_lang),
        SequenceDataset([p[1] for p in train_pairs], target_lang),
    )
    valid_dataset = ParallelDataset(
        SequenceDataset([p[0] for p in valid_pairs], source_lang),
        SequenceDataset([p[1] for p in valid_pairs], target_lang),
    )
    test_dataset = ParallelDataset(
        SequenceDataset([p[0] for p in test_pairs], source_lang),
        SequenceDataset([p[1] for p in test_pairs], target_lang),
    )
    return (train_dataset, valid_dataset, test_dataset), (source_lang, target_lang)
