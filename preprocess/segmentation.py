import collections
import io
from pathlib import Path

import nltk
from loguru import logger

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def sentence_segment(data, sentence_filter=None):
    """Segments sentences.
    
    Args:
        data: Path to an input file, string, or iterable of strings representing 
          lines of a document.
    
    Returns:
        A list of sentences.
    """
    sentences = []
    if type(data) == str:
        data = [data]
    
    if not isinstance(data, collections.Iterable):
        raise ValueError('data must be iterable')

    for line in data:
        line = line.rstrip('\n')
        if not line:
            continue
        line_sentences = nltk.tokenize.sent_tokenize(line)
        if sentence_filter is not None:
            line_sentences = list(filter(sentence_filter, line_sentences))
        sentences.extend(line_sentences)
    
    return sentences

def sentence_segment_file(path, sentence_filter=None):
    path = Path(path)
    with path.open(encoding='utf8') as f:
        return sentence_segment(f, sentence_filter=sentence_filter)