from functools import partial
from pathlib import Path

import argh
import torch
import tqdm
from loguru import logger

from .offsets import write_document_offsets
from .filtering import filter_documents, default_sentence_filter
from .intermediate import binary_indexed_dataset, to_loose_json, to_one_doc_per_line, concatenate
from .segmentation import sentence_segment_file
from .utils import parmap
from .dataset.clueweb12b import process as clueweb12b_process
from .dataset.giga5 import process as giga5_process
from .model.xlnet import xlnet

@argh.arg('--parallelize', '-p', default=1, const=None, nargs='?')
def sentence_segment(input_path, output_path, parallelize=None):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.is_dir():
        files = input_path.rglob('*')
    elif input_path.is_file():
        files = (input_path, )
    else:
        raise ValueError('Invalid input path, must be a directory or a file')

    files = [f for f in files if f.is_file()]
    
    with output_path.open('w', encoding='utf8') as output:
        for doc in parmap(sentence_segment_file, files, N=parallelize, daemon=True, progress=True):
            output.write('\n'.join(doc) + '\n\n')

@argh.arg('--parallelize', '-p', default=1, const=None, nargs='?')
def document_offsets(input_path, output_path=None, parallelize=None):
    input_path = Path(input_path)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

    if input_path.is_dir():
        files = input_path.rglob('*')
    else:
        files = (input_path, )

    files = (f for f in files if f.is_file())
    
    fn = partial(write_document_offsets, output_path=output_path)

    for _ in parmap(fn, files, N=parallelize, daemon=True, progress=True):
        pass

@argh.named('filter')
@argh.arg('--parallelize', '-p', default=1, const=None, nargs='?')
@argh.arg('--dry-run', '-d', default=None, const=-1, type=int, nargs='?')
@argh.arg('--english', dest='languages', action='store_const', const='en')
@argh.arg('--no-sent-filters', dest='sentence_filter', action='store_const', const=None)
def filter_intermediate(input_path, output_path=None, input_glob='*.txt',
                        sentence_filter=default_sentence_filter,
                        min_doc_words=100, min_doc_sents=3, min_sent_words=5,
                        languages=None, dry_run=None, parallelize=None):
    input_path = Path(input_path)
    if input_path.is_dir():
        files = input_path.rglob(input_glob)
    else:
        files = (input_path, )
    files = (f for f in files if f.is_file())

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f'output path must be a directory, {output_path} is not')
        
        output_path.mkdir(exist_ok=True, parents=True)

    if dry_run is not None:
        if dry_run >= 0:
            files = list(files)[:dry_run]
        dry_run = True
    else:
        dry_run = False

    fn = partial(filter_documents, output_path=output_path,
                 sentence_filter=sentence_filter,
                 min_doc_words=min_doc_words, min_doc_sents=min_doc_sents,
                 min_sent_words=min_sent_words, languages=languages,
                 dry_run=dry_run)

    total_in_docs = 0
    total_in_len = 0
    total_out_docs = 0
    total_out_len = 0
    for stats in parmap(fn, files, N=parallelize, daemon=True, progress=True):
        total_in_docs += stats[0]
        total_in_len += stats[1]
        total_out_docs += stats[2]
        total_out_len += stats[3]

    logger.info(
        f'Filtering Results\n'
        f'    input documents     {total_in_docs:,}\n'
        f'    output documents    {total_out_docs:,}\n'
        f'    filtered documents  {total_in_docs-total_out_docs:,} ({(total_in_docs-total_out_docs)/total_in_docs*100:.2f}%)\n\n'
        f'    input length        {total_in_len:,}\n'
        f'    output length       {total_out_len:,}\n'
        f'    filtered length     {total_in_len-total_out_len:,} ({(total_in_len-total_out_len)/total_in_len*100:.2f}%)\n'
    )

# TODO: (bnorick) enable filter_intermediate to take one large file and process
#       it in parallel by reading from offsets into the file

def concatenate_intermediate(input_path, output_path=None, input_glob='*.txt',
                             overwrite=False, dry_run=False):
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise ValueError(f'input path must be a directory, {input_path} is not')
    
    files = [f for f in input_path.rglob(input_glob) if f.is_file()]

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            raise ValueError(f'output path {output_path} exists, specify'
                              '--overwrite to overwrite it')

    concatenate(files, output_path=output_path, dry_run=dry_run)

_default_tokenizer_hub_args = ('huggingface/transformers', 'tokenizer',
                               'bert-large-cased')
@argh.arg('--tokenizer-hub-args', '-t', nargs=3,
          help='args to load tokenizer from torch.hub, e.g., ' +  \
               ' '.join(_default_tokenizer_hub_args))
@argh.arg('--parallelize', '-p', default=1, const=None, nargs='?')
def binarize(input_path, output_path=None, tokenizer_hub_args=None, input_glob='*.txt',
             dataset_impl='mmap', parallelize=None):
    input_path = Path(input_path)
    if input_path.is_dir():
        files = input_path.rglob(input_glob)
    else:
        files = (input_path, )
    files = (f for f in files if f.is_file())

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f'output path must be a directory, {output_path} is not')
        
        output_path.mkdir(exist_ok=True, parents=True)

    if tokenizer_hub_args is None:
        raise ValueError('must specify tokenizer hub args using --tokenizer-hub-args, ' +
                         'e.g., --tokenizer-hub-args ' +
                         ' '.join(_default_tokenizer_hub_args))

    # load the tokenizer so it's pre-downloaded
    tokenizer = torch.hub.load(*tokenizer_hub_args)

    fn = partial(binary_indexed_dataset, tokenizer_hub_args=tokenizer_hub_args,
                 dataset_impl=dataset_impl, output_path=output_path)

    for data_file, index_file in parmap(fn, files, N=parallelize, daemon=True, progress=True):
        pass


@argh.arg('--parallelize', '-p', default=1, const=None, nargs='?')
def intermediate_to_loose_json(input_path, output_path=None, input_glob='*.txt',
                          parallelize=None):
    """Convert from intermediate format to loose json format, retaining newlines
    between sentences."""
    input_path = Path(input_path)
    if input_path.is_dir():
        files = input_path.rglob(input_glob)
    else:
        files = (input_path, )
    files = list(f for f in files if f.is_file())

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f'output path must be a directory, {output_path} is not')
        
        output_path.mkdir(exist_ok=True, parents=True)
    
    logger.info(f'Converting {len(files)} input files from {input_path}')
    logger.info(f'Writing output to {output_path}')

    fn = partial(to_loose_json, output_path=output_path)

    for loose_json_file in parmap(fn, files, N=parallelize, daemon=True, progress=True):
        pass


@argh.arg('--parallelize', '-p', default=1, const=None, nargs='?')
def intermediate_to_one_doc_per_line(input_path, output_path=None, input_glob='*.txt',
                          parallelize=None):
    """Convert from intermediate format to one doc per line format, contiguous whitespace
    is replaced with a single space."""
    input_path = Path(input_path)
    if input_path.is_dir():
        files = input_path.rglob(input_glob)
    else:
        files = (input_path, )
    files = list(f for f in files if f.is_file())

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f'output path must be a directory, {output_path} is not')
        
        output_path.mkdir(exist_ok=True, parents=True)
    
    logger.info(f'Converting {len(files)} input files from {input_path}')
    logger.info(f'Writing output to {output_path}')

    fn = partial(to_one_doc_per_line, output_path=output_path)

    for output_file in parmap(fn, files, N=parallelize, daemon=True, progress=True):
        pass


@argh.named('clueweb12b')
@argh.arg('--parallelize', '-p', default=1, const=None, nargs='?')
@argh.arg('--no-fallback', dest='fallback', action='store_false')
def intermediate_clueweb12b(input_path, output_path, overwrite=False, parallelize=None, fallback=True):
    input_path = Path(input_path)
    if input_path.is_dir():
        files = input_path.rglob('*.warc*')
        if not files:
            raise ValueError(f'no warc files found below directory {input_path}')
    elif input_path.is_file():
        files = [input_path]
    else:
        raise ValueError(f'path must be a directory or a file, {input_path} is not')

    output_path = Path(output_path)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f'output path must be a directory, {output_path} is not')
    
    output_path.mkdir(exist_ok=True, parents=True)
    if not overwrite:
        try:
            _ = next(output_path.iterdir())
            raise ValueError(f'non-empty output path, add --overwrite to overwrite')
        except StopIteration:
            pass

    fn = partial(clueweb12b_process, output_path=output_path, fallback=fallback)
    for _ in parmap(fn, files, N=parallelize, daemon=True, progress=True):
        pass

@argh.named('giga5')
@argh.arg('--parallelize', '-p', default=1, const=None, nargs='?')
def intermediate_giga5(path, output_path, dtd_path=None, overwrite=False, parallelize=None):
    input_path = Path(input_path)
    if input_path.is_dir():
        files = input_path.rglob('*.gz')
        if not files:
            raise ValueError(f'no gzip files found below directory {input_path}')
    elif input_path.is_file():
        files = [input_path]
    else:
        raise ValueError(f'path must be a directory or a file, {input_path} is not')

    output_path = Path(output_path)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f'output path must be a directory, {output_path} is not')
    
    output_path.mkdir(exist_ok=True, parents=True)
    if not overwrite:
        try:
            _ = next(output_path.iterdir())
            raise ValueError(f'non-empty output path, add --overwrite to overwrite')
        except StopIteration:
            pass

    fn = partial(giga5_process, output_path=output_path)
    for _ in parmap(fn, files, N=parallelize, daemon=True, progress=True):
        pass

p = argh.ArghParser(prog='python -m preprocess', description='Preprocessing utilities for assorted input datasets and models')
argh.add_commands(p, [xlnet], namespace='build-model-input')
argh.add_commands(p, [document_offsets, sentence_segment, filter_intermediate, concatenate_intermediate, binarize, intermediate_to_loose_json, intermediate_to_one_doc_per_line])
argh.add_commands(p, [intermediate_clueweb12b, intermediate_giga5], namespace='build-intermediate', description='Build intermediate format from various input datasets')
argh.dispatch(p)
