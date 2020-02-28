import random
import re
from collections import namedtuple
from functools import partial
from pathlib import Path

from loguru import logger

from .vendor import langid


class FilterRule:
    def __init__(self, n):
        self.n = n
        self.count = 0
        self.sentences = []
    
    def add(self, sentence):
        if self.n is None or self.count < self.n:
            self.count += 1
            self.sentences.append(sentence)
        else:
            self.count += 1
            m = random.randint(0, self.count)
            if m < self.n:
                self.sentences[m] = sentence


class FilteredSentences:
    def __init__(self, max_sents):
        self.web_address = FilterRule(max_sents)
        self.punctuation = FilterRule(max_sents)
        self.min_words = FilterRule(max_sents)


_web_address = re.compile(r'(https?|file)://')
_punctuation = re.compile(r'[!"#$%&\\\'()*+,-./:;<=>?@[\]^_`{|}~]')
def default_sentence_filter(line, min_sent_words=5, filtered=None):
    if _web_address.search(line):
        if filtered is not None:
            filtered.web_address.add(line)
        return False
    if len(_punctuation.findall(line)) > len(line) * 0.4:
        if filtered is not None:
            filtered.punctuation.add(line)
        return False
    if len(line.split(' ')) < min_sent_words:
        if filtered is not None:
            filtered.min_words.add(line)
        return False
    return True

def filtered_documents_path(input_path, output_path):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'invalid input path {input_path}, must be a file')
    
    if output_path is None:
        return input_path.with_name(input_path.stem + '.filtered.txt')
    else:
        return Path(output_path) / (input_path.stem + '.filtered.txt')

class DryRun:
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        pass

    def write(self, data):
        pass

def filter_documents(input_path, output_path,
                     sentence_filter=default_sentence_filter,
                     min_doc_words=100, min_doc_sents=3, min_sent_words=5,
                     languages=None, dry_run=False):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'invalid input path {input_path}, must be a file')

    output_file = filtered_documents_path(input_path, output_path)

    filtered_sentences = FilteredSentences(None) if dry_run else None

    if sentence_filter is default_sentence_filter:
        filter_fn = partial(sentence_filter,
                            min_sent_words=min_sent_words,
                            filtered=filtered_sentences)
    else:
        filter_fn = sentence_filter

    languages = set(languages.split(',')) if languages is not None else None

    out_context = output_file.open('w', encoding='utf8') if not dry_run else DryRun()

    with out_context as out_f: 
        with input_path.open(encoding='utf8') as in_f:
            in_docs = 0
            in_len = 0
            out_docs = 0
            out_len = 0

            document = []
            for line in in_f:
                in_len += len(line)
                line = line.rstrip('\n')
                if not line:
                    in_docs += 1
                    if filter_fn is not None:
                        document = list(filter(filter_fn, document))
                    doc_words = sum(len(l.split(' ')) for l in document)
                    doc_sents = len(document)
                    if doc_words >= min_doc_words and doc_sents >= min_doc_sents:
                        out = '\n'.join(document) + '\n\n'
                        write = True
                        if languages is not None:
                            lang, conf = langid.classify(out)
                            if lang not in languages:
                                write = False
                        
                        if write:
                            out_f.write(out)
                            out_docs += 1
                            out_len += len(out)
                    document = []
                    continue
                document.append(line)
            
            if document:
                in_docs += 1
                if filter_fn is not None:
                    document = list(filter(filter_fn, document))
                doc_words = sum(len(l.split(' ')) for l in document)
                doc_sents = len(document)
                if doc_words >= min_doc_words and doc_sents >= min_doc_sents:
                    out = '\n'.join(document) + '\n\n'
                    write = True
                    if languages is not None:
                        lang, conf = langid.classify(out)
                        if lang not in languages:
                            write = False
                    
                    if write:
                        out_f.write(out)
                        out_docs += 1
                        out_len += len(out)
        
    if dry_run:
        logger.info((
            'Filtered sentences sample\n\n'
            'Web Address\n{}\n\n'
            'Punctuation\n{}\n\n'
            'Minimum Words\n{}\n\n').format(
                '\n'.join(filtered_sentences.web_address.sentences),
                '\n'.join(filtered_sentences.punctuation.sentences),
                '\n'.join(filtered_sentences.min_words.sentences)
            )
        )

    return in_docs, in_len, out_docs, out_len

if __name__ == '__main__':
    import sys
    document_offsets(sys.argv[1])
