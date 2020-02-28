import json
import re
from collections import namedtuple
from pathlib import Path

import torch
from tqdm import tqdm
from loguru import logger

from .utils.data import indexed_dataset

SPACE_NORMALIZER = re.compile(r"\s+")

def indexed_dataset_path(input_path, output_path, offset, suffix):
    assert suffix in {'bin', 'idx'}

    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'invalid input path {input_path}, must be a file')

    filename = '{}{}.{}'.format(
        input_path.stem,
        '.' + str(offset) if offset else '',
        suffix
    )

    if output_path is None:
        return input_path.with_name(filename)
    else:
        output_path = Path(output_path)
        if not output_path.is_dir():
            raise ValueError(f'invalid output path {output_path}, must be a directory if specified')
        return output_path / filename

def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins

BinarizeResult = namedtuple('BinarizeResult', 'nseq,ntok')

def binarize(filename, tokenizer, consumer, offset=0, end=-1):
    nseq, ntok = 0, 0

    with open(filename, 'r', encoding='utf-8') as f:
        f.seek(offset)
        # next(f) breaks f.tell(), hence readline() must be used
        line = safe_readline(f)
        while line:
            if end > 0 and f.tell() > end:
                break
            line = SPACE_NORMALIZER.sub(" ", line)
            line = line.strip()
            ids = tokenizer.encode(line)
            nseq += 1
            ntok += len(ids)
            consumer(ids)
            line = f.readline()
    return BinarizeResult(nseq=nseq, ntok=ntok)

def binary_indexed_dataset(input_path, output_path, tokenizer_hub_args,
                           offset=None, dataset_impl='mmap'):
    tokenizer = torch.hub.load(*tokenizer_hub_args)
    logger.info(f'Loading tokenizer from torch.hub:'
                f' {", ".join(tokenizer_hub_args)}')
    logger.info(f"Vocab: {tokenizer.vocab_size} tokens")

    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'invalid input path {input_path}, must be a file')

    output_path = Path(output_path)
    data_file = indexed_dataset_path(input_path, output_path, offset, 'bin')
    index_file = indexed_dataset_path(input_path, output_path, offset, 'idx')

    logger.info(f'Processing {input_path}')
    logger.info(f'Data file {data_file}')
    logger.info(f'Index file {index_file}')

    ds = indexed_dataset.make_builder(data_file,
                                      impl=dataset_impl,
                                      vocab_size=len(tokenizer.vocab))

    res = binarize(input_path, tokenizer, lambda t: ds.add_item(torch.tensor(t)))

    sentence_count = res.nseq
    token_count = res.ntok

    ds.finalize(index_file)

    logger.info(
        "{}: {} sents, {} tokens".format(input_path, sentence_count,
                                         token_count)
    )

    return data_file, index_file

def to_loose_json(input_path, output_path):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'invalid input path {input_path}, must be a file')

    input_name = input_path.stem

    output_path = Path(output_path)
    loose_json_file = output_path /  input_path.with_suffix('.ljson').name
    
    doc_count = 0
    empty_doc_count = 0
    with input_path.open(encoding='utf-8') as f_in:
        with loose_json_file.open('w', encoding='utf8') as f_out:
            doc = []
            doc_start = f_in.tell()
            line = safe_readline(f_in)
            while line:
                line = line.rstrip('\n')
                if line:
                    doc.append(line)
                else:
                    if doc:
                        out_doc = {'text': '\n'.join(doc), 'id': f'{input_name}-{doc_start}'}
                        f_out.write(json.dumps(out_doc)+'\n')
                        doc_count += 1
                        doc = []
                        doc_start = f_in.tell()
                    else:
                        empty_doc_count += 1

                line = f_in.readline()

            if doc:
                out_doc = {'text': '\n'.join(doc), 'id': f'{input_name}-{doc_start}'}
                f_out.write(json.dumps(out_doc)+'\n')
        
    logger.info(
        "{}: {} non-empty docs, {} empty docs".format(input_path, doc_count,
                                         empty_doc_count)
    )

    return loose_json_file

def to_one_doc_per_line(input_path, output_path):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'invalid input path {input_path}, must be a file')

    input_name = input_path.stem

    output_path = Path(output_path)
    output_file = output_path /  input_path.with_suffix('.one_doc_per_line.txt').name
    
    doc_count = 0
    empty_doc_count = 0
    with input_path.open(encoding='utf-8') as f_in:
        with output_file.open('w', encoding='utf8') as f_out:
            doc = []
            line = safe_readline(f_in)
            while line:
                line = line.rstrip('\n')
                clean_line = SPACE_NORMALIZER.sub(' ', line)
                if line:
                    if clean_line:
                        doc.append(clean_line)
                else:
                    if doc:
                        f_out.write(' '.join(doc)+'\n')
                        doc_count += 1
                        doc = []
                    else:
                        empty_doc_count += 1

                line = f_in.readline()

            if doc:
                f_out.write(' '.join(doc)+'\n')
        
    logger.info(
        "{}: {} non-empty docs, {} empty docs".format(input_path, doc_count,
                                         empty_doc_count)
    )

    return output_file

def concatenate(files, output_path=None, overwrite=False, dry_run=False):
    files = [f for f in files if f.stat().st_size > 0]
    if not files:
        raise ValueError(f'only empty files passed to concatenate')

    if output_path is None:
        parents = set(files[0].parents)
        for f in files[1:]:
            parents &= set(f.parents)
        parents = [(len(p.parts), p) for p in parents]
        output_path = sorted(parents)[-1][1] / 'combined.txt'
        if output_path.is_file() and not overwrite:
            raise ValueError(f'output path {output_path} exists, specify'
                             f'--overwrite to overwrite it')
    
    logger.info(f'Combining {len(files)} documents into {output_path}')
    if not dry_run:
        with output_path.open('w', encoding='utf8', newline='\n') as f_out:
            for file_path in tqdm(files, desc='files'):
                with file_path.open('r', encoding='utf8', newline='\n') as f_in:
                    data = f_in.read()

                f_out.write(data)
                if data[-1] != '\n':
                    f_out.write('\n\n')
                elif data[-2] != '\n':
                    f_out.write('\n')
    logger.info('Finished')