import argparse
import hashlib
import itertools
import json
import logging
import multiprocessing
import random
import shutil
import sys
import unicodedata
import uuid
from functools import partial
from pathlib import Path

import argh
import numpy as np
import sentencepiece as spm
from loguru import logger
from tqdm import tqdm, trange

SPIECE_UNDERLINE = '▁'

DATA = None
SENT_IDS = None

_SPECIAL_SYMBOLS = {
    "<unk>"  : 0,
    "<s>"    : 1,
    "</s>"   : 2,
    "<cls>"  : 3,
    "<sep>"  : 4,
    "<pad>"  : 5,
    "<mask>" : 6,
    "<eod>"  : 7,
    "<eop>"  : 8,
}

CLS_ID = _SPECIAL_SYMBOLS["<cls>"]
SEP_ID = _SPECIAL_SYMBOLS["<sep>"]

# generate vocab
# for each file, tokenize and make numpy arrays for data and sent_ids
# combine all the files (shuffled based on pass) into one data and send_ids

def parse_args(args=None, allow_unknown=False):
    parser = argparse.ArgumentParser('XLNet Data Preprocessing')
    parser.add_argument('--sentencepiece-path', '-sp', required=True, type=str, help='path to sentencepiece model file')
    parser.add_argument('--input-path', '-i', required=True, type=str, help='directory containing input files')
    parser.add_argument('--input-glob', '-g', type=str, default='*.txt', help='glob to match input files')
    parser.add_argument('--output-path', '-o', required=True, type=str, help='directory to put output')
    parser.add_argument('--parallelize', '-p', type=int, nargs='?', const=-1, default=None, help='level of parallelization (i.e., number of processes), default is not to parallelize and passing --parallelize without a value results in to one process per core')
    parser.add_argument('--pass-id', '-pi', type=int, default=0, help='pass over the data, results in random file-level shuffling and negatives')

    parser.add_argument('--uncased', '-u', action='store_true', help='preprocess data to remove case')
    parser.add_argument('--bidirectional', '-b', action='store_true', help='generate both forward and backward samples')
    parser.add_argument('--reuse-len', '-r', type=int, default=256, help='how much of each sequence to reuse in next sample')
    parser.add_argument('--seq-len', '-s', type=int, default=512, help='sequence length for each sample')
    parser.add_argument('--mask', '-m', type=int, default=85, help='number of tokens to mask')
    parser.add_argument('--mask-alpha', '-ma', type=int, default=6, help='how many tokens per group')
    parser.add_argument('--mask-beta', '-mb', type=int, default=1, help='how many tokens to mask per group')

    return parser.parse_args(args) if not allow_unknown else parser.parse_known_args(args)

def _is_start_piece(piece):
    special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
    if (piece.startswith(SPIECE_UNDERLINE) or piece.startswith("<")
            or piece in special_pieces):
        return True
    else:
        return False

def encode_pieces(sp_model, text, sample=False):
    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ''))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    return new_pieces

def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids

def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
    if remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace("``", '"').replace("''", '"')

    if not keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])

    if lower:
        outputs = outputs.lower()

    return outputs

def _load_data_from_paths(input_paths, intermediate_path, sentencepiece_path, uncased):
    # Load sentence-piece model
    sp = spm.SentencePieceProcessor()
    sp.Load(sentencepiece_path)

    if type(input_paths) != list:
        input_paths = [input_paths]
    input_paths = [Path(p) for p in input_paths]

    logger.info(f"Got {len(input_paths)} files ({', '.join(p.name for p in input_paths)})")

    input_shards = []
    total_line_cnt = 0
    for input_path in input_paths:
        input_data, sent_ids, doc_offsets = [], [], [0]
        sent_id, line_cnt, cur_offset = True, 0, 0
        logger.info(f"Processing {input_path}")
        for line in input_path.open(encoding='utf8'):
            if line_cnt % 100000 == 0:
                logger.info(f"Loading line {line_cnt}")
            line_cnt += 1
            if not line.strip():
                doc_offsets.append(cur_offset)
                continue
            else:
                cur_sent = preprocess_text(line.strip(), lower=uncased)
                cur_sent = encode_ids(sp, cur_sent)

            input_data.extend(cur_sent)
            sent_ids.extend([sent_id] * len(cur_sent))
            sent_id = not sent_id
            cur_offset += len(cur_sent)
        
        # if file didn't end in newline, add the last offset value
        if doc_offsets[-1] != cur_offset:
            doc_offsets.append(cur_offset)

        logger.info(f"Finish with line {line_cnt}")
        if line_cnt == 0:
            continue

        input_data = np.array(input_data, dtype=np.int64)
        sent_ids = np.array(sent_ids, dtype=np.bool)
        doc_offsets = np.array(doc_offsets, dtype=np.int64)

        total_line_cnt += line_cnt
        intermediate_file = (intermediate_path / f'{input_path.stem}.npz')
        np.savez(str(intermediate_file), data=input_data, sent_ids=sent_ids, doc_offsets=doc_offsets)
        input_shards.append(intermediate_file)
        # input_shards.append((input_data, sent_ids))

    logger.info(f"Total number line: {total_line_cnt}")

    return input_shards

def save_args(path, args):
    with path.open('w', encoding='utf8') as f:
        d = vars(args)
        # remove argh stuff
        d.pop('_functions_stack', None)
        json.dump(d, f, indent=2, sort_keys=True)

def load_data(args):
    logger.info('load_data called')

    pool = None
    if args.parallelize is not None:
        processes = None if args.parallelize == -1 else args.parallelize
        pool = multiprocessing.Pool(processes=processes)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    corpus_info_path = output_path / 'corpus_info.json'

    corpus_info = {}
    if corpus_info_path.is_file():
        with corpus_info_path.open(encoding='utf8') as f:
            corpus_info = json.load(f)

        if corpus_info['sentencepiece_path'] != args.sentencepiece_path:
            raise RuntimeError(f'Corpus at {output_path} was generated with a different sentencepiece model.\n' \
                               f'    corpus path: {corpus_info["sentencepiece_path"]}\n' \
                               f'    input path:  {args.sentencepiece_path}')

    if not hasattr(args, 'intermediate'):
        args.intermediate = {}

    all_shard_paths = []

    for input_path in sorted(args.input_paths):
        input_path = Path(input_path)
        if not input_path.is_dir():
            raise ValueError(f'invalid input path {input_path}, '
                             f'every input path must be a directory')

        input_paths = sorted(input_path.rglob(args.input_glob))
        # ignore emply files
        input_paths = [p for p in input_paths if p.stat().st_size > 0]
    
        intermediate_folder = hashlib.md5(str(input_path).encode('utf8')).hexdigest()
        if 'intermediate' in corpus_info:
            intermediate_folder = corpus_info['intermediate'].get(
                str(input_path),
                intermediate_folder
            )

        args.intermediate[str(input_path)] = intermediate_folder

        intermediate_path = output_path / 'intermediate' / intermediate_folder

        shard_paths = []

        intermediate = False
        if intermediate_path.is_dir():
            logger.info('Intermediate directory exists, attempting to load')
            intermediate = True
            missing_paths, missing_indices = [], []
            for i, input_path in enumerate(input_paths):
                intermediate_file = intermediate_path / f'{input_path.stem}.npz'
                if not intermediate_file.is_file():
                    missing_paths.append(input_path)
                    missing_indices.append(i)
                    shard_paths.append(None)
                else:
                    shard_paths.append(intermediate_file)
                    # data = np.load(intermediate_file)
                    # shard_paths.append((data['data'], data['sent_ids']))

            if missing_paths:
                all_input_paths = input_paths
                input_paths = missing_paths
                logger.info(f'Partially loaded intermediate data from {intermediate_path} [{len(all_input_paths)-len(missing_paths)}/{len(all_input_paths)} files]')
            else:
                logger.info(f'Loaded intermediate data from {intermediate_path}')
                all_shard_paths.extend(shard_paths)
                continue
        else:
            intermediate_path.mkdir(parents=True)

        fn = partial(_load_data_from_paths, intermediate_path=intermediate_path, sentencepiece_path=args.sentencepiece_path, uncased=args.uncased)

        if pool is not None:
            loaded_shard_paths = pool.map(fn, input_paths, chunksize=1)
            loaded_shard_paths = list(itertools.chain.from_iterable(loaded_shard_paths))  # flatten list of lists
        else:
            loaded_shard_paths = []
            for input_path in tqdm(input_paths, desc='input files'):
                loaded_shard_paths.extend(fn(input_path))

        if intermediate:
            # put each shard in the right location in shards
            for i, shard_path in enumerate(loaded_shard_paths):
                shard_paths[missing_indices[i]] = shard_path
        else:
            # we just loaded all the shards at once
            shard_paths = loaded_shard_paths

        assert all(p is not None for p in shard_paths)
        all_shard_paths.extend(shard_paths)

    args.run_cmd = ' '.join(sys.argv)
    save_args(corpus_info_path, args)

    return all_shard_paths

def _split_a_and_b(data, sent_ids, begin_idx, tot_len, extend_target=False):
    """Split two segments from `data` starting from the index `begin_idx`."""

    data_len = data.shape[0]
    if begin_idx + tot_len >= data_len:
        logger.info(f"[_split_a_and_b] returns None: "
                    f"begin_idx {begin_idx} + tot_len {tot_len} >= data_len {data_len}")
        return None

    end_idx = begin_idx + 1
    cut_points = []
    while end_idx < data_len:
        if sent_ids[end_idx] != sent_ids[end_idx - 1]:
            if end_idx - begin_idx >= tot_len: break
            cut_points.append(end_idx)
        end_idx += 1

    a_begin = begin_idx
    if len(cut_points) == 0 or random.random() < 0.5:
        label = 0
        if len(cut_points) == 0:
            a_end = end_idx
        else:
            a_end = random.choice(cut_points)

        b_len = max(1, tot_len - (a_end - a_begin))
        # (zihang): `data_len - 1` to account for extend_target
        b_begin = random.randint(0, data_len - 1 - b_len)
        b_end = b_begin + b_len
        while b_begin > 0 and sent_ids[b_begin - 1] == sent_ids[b_begin]:
            b_begin -= 1
        # (zihang): `data_len - 1` to account for extend_target
        while b_end < data_len - 1 and sent_ids[b_end - 1] == sent_ids[b_end]:
            b_end += 1

        new_begin = a_end
    else:
        label = 1
        a_end = random.choice(cut_points)
        b_begin = a_end
        b_end = end_idx

        new_begin = b_end

    while a_end - a_begin + b_end - b_begin > tot_len:
        if a_end - a_begin > b_end - b_begin:
            # delete the right side only for the LM objective
            a_end -= 1
        else:
            b_end -= 1

    ret = [data[a_begin: a_end], data[b_begin: b_end], label, new_begin]

    if extend_target:
        if a_end >= data_len or b_end >= data_len:
            logger.info(f"[_split_a_and_b] returns None: "
                        f"a_end {a_end} or b_end {b_end} >= data_len {data_len}")
            return None
        a_target = data[a_begin + 1: a_end + 1]
        b_target = data[b_begin: b_end + 1]
        ret.extend([a_target, b_target])

    return ret

def _sample_mask(sp, seg, mask_alpha, mask_beta, reverse=False, max_gram=5, goal_num_mask=None):
    """Sample `goal_num_mask` tokens for partial prediction.
    About `mask_beta` tokens are chosen in a context of `mask_alpha` tokens."""

    seg_len = len(seg)
    mask = np.array([False] * seg_len, dtype=np.bool)

    num_mask = 0

    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_gram + 1)
    pvals /= pvals.sum(keepdims=True)

    if reverse:
        seg = np.flip(seg, 0)

    cur_len = 0
    while cur_len < seg_len:
        if goal_num_mask is not None and num_mask >= goal_num_mask: break

        n = np.random.choice(ngrams, p=pvals)
        if goal_num_mask is not None:
            n = min(n, goal_num_mask - num_mask)
        ctx_size = (n * mask_alpha) // mask_beta
        l_ctx = np.random.choice(ctx_size)
        r_ctx = ctx_size - l_ctx

        # Find the start position of a complete token
        beg = cur_len + l_ctx
        while beg < seg_len and not _is_start_piece(sp.IdToPiece(seg[beg].item())):
            beg += 1
        if beg >= seg_len:
            break

        # Find the end position of the n-gram (start pos of the n+1-th gram)
        end = beg + 1
        cnt_ngram = 1
        while end < seg_len:
            if _is_start_piece(sp.IdToPiece(seg[beg].item())):
                cnt_ngram += 1
                if cnt_ngram > n:
                    break
            end += 1
        if end >= seg_len:
            break

        # Update
        mask[beg:end] = True
        num_mask += end - beg

        cur_len = end + r_ctx

    while goal_num_mask is not None and num_mask < goal_num_mask:
        i = np.random.randint(seg_len)
        if not mask[i]:
            mask[i] = True
            num_mask += 1

    if reverse:
        mask = np.flip(mask, 0)

    return mask

def _make_samples(start, end, reverse, worker_idx, tmp_path, sentencepiece_path, seq_len, reuse_len, bidir, mask_alpha, mask_beta, mask=None, data=None, sent_ids=None):
    logger.info(f'making samples from {start} to {end}')

    sp = spm.SentencePieceProcessor()
    sp.Load(sentencepiece_path)

    id_type = np.uint16 if len(sp) <= 2**16 else np.uint32
    Sample = np.dtype(
        [
            ('input', id_type, seq_len),
            ('is_masked', np.bool, seq_len),
            ('target', id_type, seq_len),
            ('seg_id', np.int8, seq_len),
            ('label', np.int8)
        ]
    )

    if data is None:
        data = DATA
    if sent_ids is None:
        sent_ids = SENT_IDS

    if reverse:
        data = data[::-1]
        sent_ids = sent_ids[::-1]

    samples_created = 0

    sep_array = np.array([SEP_ID], dtype=np.int64)
    cls_array = np.array([CLS_ID], dtype=np.int64)

    sample_count = (end-start)//reuse_len
    samples = np.empty(sample_count, dtype=Sample)

    for sample, curr in enumerate(range(start, end, reuse_len)):
        if sample and sample % (sample_count // 4) == 0:
            logger.info(f'processed {sample}')
        inp = data[curr: curr + reuse_len]
        tgt = data[curr + 1: curr + reuse_len + 1]

        results = _split_a_and_b(
            data,
            sent_ids,
            begin_idx=curr + reuse_len,
            tot_len=seq_len - reuse_len - 3,
            extend_target=True)
        if results is None:
            logger.info(f"Break out with seq idx {curr}")
            break

        (a_data, b_data, label, _, a_target, b_target) = tuple(results)
        if mask is None:
            num_mask_0 = num_mask_1 = None
        else:
            num_mask_1 = mask // 2
            num_mask_0 = mask - num_mask_1
        mask_0 = _sample_mask(sp, inp, mask_alpha, mask_beta, reverse=reverse,
                              goal_num_mask=num_mask_0)
        mask_1 = _sample_mask(sp, np.concatenate([a_data, sep_array, b_data,
                                                  sep_array, cls_array]),
                              mask_alpha, mask_beta,
                              reverse=reverse, goal_num_mask=num_mask_1)

        # concatenate data
        cat_data = np.concatenate([inp, a_data, sep_array, b_data,
                                   sep_array, cls_array])
        seg_id = np.array([0] * (reuse_len + a_data.shape[0]) + [0] +
                            [1] * b_data.shape[0] + [1] + [2])
        assert cat_data.shape[0] == seq_len
        assert seg_id.shape[0] == seq_len
        assert mask_0.shape[0] == seq_len // 2
        assert mask_1.shape[0] == seq_len // 2

        # the last two CLS's are not used, just for padding purposes
        tgt = np.concatenate([tgt, a_target, b_target, cls_array, cls_array])
        assert tgt.shape[0] == seq_len

        is_masked = np.concatenate([mask_0, mask_1], 0)
        if mask is not None:
            assert np.sum(is_masked) == mask

        samples[sample]['input'] = cat_data
        samples[sample]['is_masked'] = is_masked
        samples[sample]['target'] = tgt
        samples[sample]['seg_id'] = seg_id
        samples[sample]['label'] = label

        samples_created += 1

    logger.info(f"done sampling    sample_count {sample_count}    samples_created {samples_created}")

    if tmp_path is not None:
        samples_path = tmp_path / f'{worker_idx}.npy'
        logger.info(f'writing samples to {samples_path}')
        np.save(samples_path, samples)
        return samples_path
    else:
        return samples


def _load_samples(sample_paths):
    return [np.load(sample_path) for sample_path in sample_paths]

def generate_samples(args, shard_paths):
    logger.info('generate_samples called')

    np.random.seed(100 + args.pass_id)
    random.seed(100 + args.pass_id)

    logger.info('loading shards')
    data = []
    sent_ids = []
    shard_offset = 0
    doc_offsets = []
    for shard_path in tqdm(shard_paths, desc='shards'):
        shard_data = np.load(str(shard_path))
        data.append(shard_data['data'])
        sent_ids.append(shard_data['sent_ids'])
        
        offsets_start_idx = 0 if shard_offset == 0 else 1
        doc_offsets.append(shard_data['doc_offsets'][offsets_start_idx:] + shard_offset)
        shard_offset += shard_data['data'].shape[0]

    data = np.concatenate(data)
    sent_ids = np.concatenate(sent_ids)
    doc_offsets = np.concatenate(doc_offsets)[:, np.newaxis] # as a column

    logger.info(f'shuffling all {len(doc_offsets)} docs')
    # start index is first column, end index is second column
    doc_boundaries = np.hstack([doc_offsets[:-1], doc_offsets[1:]])
    np.random.shuffle(doc_boundaries)
    doc_indices = np.concatenate([np.arange(s,e) for s,e in doc_boundaries])
    data = data[doc_indices]
    sent_ids = sent_ids[doc_indices]
    
    # import code; code.interact(local=locals())

    logger.info('fixing sent ids')
    prev_sent_id = None
    pos = 0
    invert = []
    for start_idx, end_idx in doc_boundaries:
        length = end_idx - start_idx
        if sent_ids[pos] == prev_sent_id:
            invert.append(np.arange(pos, pos + length))
            prev_sent_id = not sent_ids[pos + length - 1]
        else:
            prev_sent_id = sent_ids[pos + length - 1]
        pos += length

    np.logical_not.at(sent_ids, np.concatenate(invert))

    sample_count = (len(data) - args.seq_len) // args.reuse_len + 1
    cut_idx = (sample_count - 1) * args.reuse_len + args.seq_len
    data = data[:cut_idx]
    sent_ids = sent_ids[:cut_idx]

    # use globals for copy-on-write in multiprocessing
    global DATA, SENT_IDS
    DATA = data
    SENT_IDS = sent_ids

    # make the pool after globals are set for copy-on-write
    pool = None
    if args.parallelize is not None:
        processes = None if args.parallelize == -1 else args.parallelize
        pool = multiprocessing.Pool(processes=processes)

    if pool is not None:
        tmp_path = Path(f'/dev/shm/{uuid.uuid4()}')
        tmp_path.mkdir()
    fn = partial(_make_samples,
                 tmp_path=tmp_path,
                 sentencepiece_path=args.sentencepiece_path,
                 seq_len=args.seq_len,
                 reuse_len=args.reuse_len,
                 bidir=args.bidirectional,
                 mask=args.mask,
                 mask_alpha=args.mask_alpha,
                 mask_beta=args.mask_beta
    )


    if pool is not None:
        num_workers = args.parallelize if args.parallelize != -1 else multiprocessing.cpu_count()
        samples_per_worker = sample_count // num_workers
        splits = [0]
        for i in range(num_workers):
            worker_samples = samples_per_worker + (1 if i < sample_count % num_workers else 0)
            splits.append(splits[-1] + worker_samples * args.reuse_len)
        assert len(splits) == num_workers + 1

        logger.info('generating forward samples')
        samples = pool.starmap(fn, zip(splits[:-1], splits[1:], [False] * num_workers, range(num_workers)), chunksize=1)  # [0, 4, 8, 12] => [(0, 4), (4, 8), (8, 12)]
        if tmp_path is not None:
            samples = _load_samples(samples)
        logger.info('concatenating forward samples')
        samples = np.concatenate(samples)
        if args.bidirectional:
            fw_samples = samples
            logger.info('generating backward samples')
            bw_samples = pool.starmap(fn, zip(splits[:-1], splits[1:], [True] * num_workers, range(num_workers)), chunksize=1)
            if tmp_path is not None:
                bw_samples = _load_samples(bw_samples)
            logger.info('concatenating backward samples')
            bw_samples = np.concatenate(bw_samples)
    else:
        end = sample_count * args.reuse_len + args.seq_len
        logger.info('generating forward samples')
        samples = fn(0, end, False, -1)
        if args.bidirectional:
            fw_samples = samples
            logger.info('generating backward samples')
            bw_samples = fn(0, end, True, -1)

    logger.info(f'generate_samples done    sample_count {sample_count}    samples {samples.shape[0]}')
    if tmp_path is not None:
        logger.info(f'removing shared path {tmp_path}')
        shutil.rmtree(str(tmp_path))

    if args.bidirectional:
        return fw_samples, bw_samples
    else:
        return samples

def directory_name_from_args(args, prefix='', suffix=''):
    return format_directory_name(args.seq_len, args.reuse_len, args.uncased,
                                 args.mask_alpha, args.mask_beta, args.mask,
                                 prefix=prefix, suffix=suffix)

def format_directory_name(seq_len, reuse_len, uncased, mask_alpha,
                          mask_beta, mask, prefix='', suffix=''):
    if prefix:
        prefix += '.'
    if suffix and not suffix[0] == '.':
        suffix = '.' + suffix

    uncased_str = '' if not uncased else 'uncased.'

    dir_name = '{}seqlen-{}.reuse-{}.{}alpha-{}.beta-{}.masked-{}{}'.format(
        prefix, seq_len, reuse_len, uncased_str,
        mask_alpha, mask_beta, mask, suffix)

    return dir_name

def save_samples(args, samples, direction='fw'):
    output_path = Path(args.output_path)
    output_dir = output_path / directory_name_from_args(args)
    output_dir.mkdir(exist_ok=True, parents=True)

    save_args(output_dir / 'args.json', args)

    if type(samples) == tuple:
        output = zip(('fw', 'bw'), samples)
    else:
        output = [(direction, samples)]

    for direction, samples in output:
        file_path = output_dir / f'{direction}_samples.npy'
        logger.info(f'saving {file_path}')
        np.save(str(file_path), samples)

# def main(sentencepiece_path, input_path, output_path, input_glob='*.txt',
#          parallelize=None, pass_id=0, uncased=False, bidirectional=True,
#          reuse_len=256, seq_len=512, mask=85, mask_alpha=6, mask_beta=1):
#    pass

@argh.arg('--sentencepiece-path', '-sp', required=True, type=str, help='path to sentencepiece model file')
@argh.arg('--input-paths', '-i', required=True, nargs='+', type=str, help='directory containing input files')
@argh.arg('--input-glob', '-g', type=str, default='*.txt', help='glob to match input files')
@argh.arg('--output-path', '-o', required=True, type=str, help='directory to put output')
@argh.arg('--parallelize', '-p', type=int, nargs='?', const=-1, default=None, help='level of parallelization (i.e., number of processes), default is not to parallelize and passing --parallelize without a value results in to one process per core')
@argh.arg('--pass-id', '-pi', type=int, default=0, help='pass over the data, results in random file-level shuffling and negatives')
@argh.arg('--uncased', '-u', action='store_true', help='preprocess data to remove case')
@argh.arg('--bidirectional', '-b', action='store_true', help='generate both forward and backward samples')
@argh.arg('--reuse-len', '-r', type=int, default=256, help='how much of each sequence to reuse in next sample')
@argh.arg('--seq-len', '-s', type=int, default=512, help='sequence length for each sample')
@argh.arg('--mask', '-m', type=int, default=85, help='number of tokens to mask')
@argh.arg('--mask-alpha', '-ma', type=int, default=6, help='how many tokens per group')
@argh.arg('--mask-beta', '-mb', type=int, default=1, help='how many tokens to mask per group')
@argh.expects_obj
def xlnet(args):
    shard_paths = load_data(args)
    samples = generate_samples(args, shard_paths)
    save_samples(args, samples)

    logger.info('done')

# if __name__ == '__main__':
#     args = parse_args()
#     shard_paths = load_data(args)
#     samples = generate_samples(args, shard_paths)
#     save_samples(args, samples)

#     logger.info('done')
