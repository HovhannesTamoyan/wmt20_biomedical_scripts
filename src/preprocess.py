#!/usr/bin/env python

import os
import logging
import argparse
import requests
import itertools

from typing import Set, List
from glob import glob
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


SUBSETS = ('test', 'valid', 'train')

PREDEFINED_VOCAB_MODELS = {
    ('en', 'de'): '/pretrained/wmt19.en-de.joined-dict.single_model',
    ('de', 'en'): '/pretrained/wmt19.de-en.joined-dict.single_model',
    ('en', 'ru'): '/pretrained/wmt19.en-ru.single_model',
    ('ru', 'en'): '/pretrained/wmt19.ru-en.single_model',
}

CLEAN_SCRIPT = os.path.join(os.path.dirname(__file__), "mosesdecoder/scripts/training/clean-corpus-n.perl")


def prep(input_path: str,
         output_path: str,
         lang: str,
         vocab_model: str,
         num_workers: int = 1,
         fusmc: bool = False):

    bpecodes_path = os.path.join(vocab_model, f"bpecodes.{lang}")
    dict_path = os.path.join(vocab_model, f"dict.{lang}.txt")

    normalize_extra_args = ''
    tokenize_extra_args = ''

    if fusmc:
        normalize_extra_args = ' --normalize-quote-commas --normalize-numbers '
        tokenize_extra_args = ' -x '


    os.system(f"""cat "{input_path}" \
        | sacremoses -l "{lang}" -j "{num_workers}" --quiet \
          normalize {normalize_extra_args} \
          --replace-unicode-puncts \
          --remove-control-chars \
        | sacremoses -l "{lang}" -j "{num_workers}" \
          tokenize {tokenize_extra_args} \
          --aggressive-dash-splits \
        | fast applybpe_stream "{bpecodes_path}" "{dict_path}" \
        > "{output_path}" \
    """)

    return output_path


def detect_langs(input_dir: str):
    srcs: Set[str] = set()
    tgts: Set[str] = set()
    pattern = os.path.join(input_dir, "*.*-*.*")
    logger.warning(f"Searching for {pattern}")
    for path in glob(pattern):
        filename = os.path.basename(path)
        subset, pair, lang = filename.split('.')
        if subset not in SUBSETS:
            logger.warning('Found possible subset that is not `train`, `valid` or `test`')
            logger.warning('Ignoring...')
            continue
        src, tgt = pair.split('-')
        assert lang in (src, tgt)
        srcs.add(src)
        tgts.add(tgt)

    assert len(srcs) == 1
    assert len(tgts) == 1

    return src, tgt


def prep_dir(input_dir: str,
             output_dir: str,
             src: str,
             tgt: str,
             vocab_model: str,
             num_workers: int):

    os.makedirs(output_dir, exist_ok=False)

    for subset in SUBSETS:
        src_filename = f"{subset}.{src}-{tgt}.{src}"
        tgt_filename = f"{subset}.{src}-{tgt}.{tgt}"
        src_in_path = os.path.join(input_dir, src_filename)
        tgt_in_path = os.path.join(input_dir, tgt_filename)

        if not os.path.exists(src_in_path) and not os.path.exists(tgt_in_path):
            continue
        assert os.path.exists(src_in_path) and os.path.exists(tgt_in_path)

        src_out_path = os.path.join(output_dir, src_filename)
        tgt_out_path = os.path.join(output_dir, tgt_filename)

        logger.warning(f"Writing preprocessed to: {src_out_path}...")
        prep(input_path=src_in_path,
             output_path=src_out_path,
             lang=src,
             vocab_model=vocab_model,
             num_workers=num_workers)

        logger.warning(f"Writing preprocessed to: {tgt_out_path}...")
        prep(input_path=tgt_in_path,
             output_path=tgt_out_path,
             lang=tgt,
             vocab_model=vocab_model,
             num_workers=num_workers)

        logger.warning(f"")
    logger.warning(f"")

    return output_dir


def clean_pair(input_path: str,
               output_path: str,
               src: str,
               tgt: str,
               ratio: float,
               min_len: int,
               max_len: int):
    os.system(f"{CLEAN_SCRIPT} -ratio {ratio} {input_path} {src} {tgt} {output_path} {min_len} {max_len}")
    return output_path


def clean_dir(input_dir: str,
              output_dir: str,
              src: str,
              tgt: str,
              ratio: float,
              min_len: int,
              max_len: int):

    os.makedirs(output_dir, exist_ok=False)

    for subset in SUBSETS:
        filename = f"{subset}.{src}-{tgt}"
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        src_filename = f"{subset}.{src}-{tgt}.{src}"
        tgt_filename = f"{subset}.{src}-{tgt}.{tgt}"
        src_in_path = os.path.join(input_dir, src_filename)
        tgt_in_path = os.path.join(input_dir, tgt_filename)

        if not os.path.exists(src_in_path) and not os.path.exists(tgt_in_path):
            continue
        assert os.path.exists(src_in_path) and os.path.exists(tgt_in_path)

        src_out_path = os.path.join(output_dir, src_filename)
        tgt_out_path = os.path.join(output_dir, tgt_filename)

        logger.warning(f"Writing cleaned to: {src_out_path} and {tgt_out_path}")

        if subset == 'test':
            logger.warning(f"Be careful! Filtering some instances from test!")

        clean_pair(input_path=in_path,
                   output_path=out_path,
                   src=src, tgt=tgt,
                   ratio=ratio, min_len=min_len, max_len=max_len)


def binarize_dir(input_dir: str,
                 output_dir: str,
                 src: str,
                 tgt: str,
                 vocab_model: str,
                 num_workers: int):
    args = []
    for subset in SUBSETS:
        filename = f"{subset}.{src}-{tgt}"
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        src_filename = f"{subset}.{src}-{tgt}.{src}"
        tgt_filename = f"{subset}.{src}-{tgt}.{tgt}"
        src_in_path = os.path.join(input_dir, src_filename)
        tgt_in_path = os.path.join(input_dir, tgt_filename)

        if not os.path.exists(src_in_path) and not os.path.exists(tgt_in_path):
            continue
        assert os.path.exists(src_in_path) and os.path.exists(tgt_in_path)

        args.append(f"--{subset}pref")
        args.append(in_path)

    data_args = ' '.join(args)

    os.system(f"""
        fairseq-preprocess {data_args} \
        --source-lang "{src}" \
        --target-lang "{tgt}" \
        --srcdict "{vocab_model}/dict.{src}.txt" \
        --tgtdict "{vocab_model}/dict.{tgt}.txt" \
        --destdir "{output_dir}" \
        --workers "{num_workers}" \
        --multiple-files
    """)


def read_parallel(src_path: str, tgt_path: str):
    with open(src_path, 'r') as src_f, open(tgt_path, 'r') as tgt_f:
        for src_line, tgt_line in zip(src_f, tgt_f):
            src_line = src_line.rstrip('\n')
            tgt_line = tgt_line.rstrip('\n')
            if src_line is None and tgt_line is None:
                continue
            yield src_line, tgt_line


def read_parallel_batches(src_path: str, tgt_path: str, batch_size: int):
    generator = read_parallel(src_path, tgt_path)
    while True:
        src_lines: List[str] = []
        tgt_lines: List[str] = []
        for src_line, tgt_line in itertools.islice(generator, batch_size):
            src_lines.append(src_line)
            tgt_lines.append(tgt_line)

        if not src_lines and not tgt_lines:
            break

        yield src_lines, tgt_lines


class QE:

    def check(self, src_line: str, tgt_line: str) -> bool:
        raise NotImplementedError

    def check_batch(self,
                    src_lines: List[str],
                    tgt_lines: List[str]) -> List[bool]:
        results: List[bool] = []
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            results.append(self.check(src_line, tgt_line))
        return results

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return self

    def close(self):
        return


def filter_batches(src_in_path: str,
                   tgt_in_path: str,
                   qe: QE,
                   batch_size: int):
    for src_lines, tgt_lines in read_parallel_batches(src_in_path,
                                                      tgt_in_path,
                                                      batch_size):
        statuses = qe.check_batch(src_lines, tgt_lines)
        yield from zip(src_lines, tgt_lines, statuses)


def filter(src_in_path: str,
           tgt_in_path: str,
           src_out_path: str,
           tgt_out_path: str,
           qe: QE,
           batch_size: int = 15):
    filtered = filter_batches(src_in_path, tgt_in_path, qe=qe, batch_size=batch_size)

    with open(src_out_path, 'w') as src_w, open(tgt_out_path, 'w') as tgt_w:
        for src_line, tgt_line, status in tqdm(filtered):
            if not status:
                continue
            src_w.write(f"{src_line}\n")
            tgt_w.write(f"{tgt_line}\n")


def filter_dir(input_dir: str,
               output_dir: str,
               src: str,
               tgt: str,
               qe: QE):

    os.makedirs(output_dir, exist_ok=False)

    for subset in SUBSETS:
        filename = f"{subset}.{src}-{tgt}"
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        src_filename = f"{subset}.{src}-{tgt}.{src}"
        tgt_filename = f"{subset}.{src}-{tgt}.{tgt}"
        src_in_path = os.path.join(input_dir, src_filename)
        tgt_in_path = os.path.join(input_dir, tgt_filename)

        if not os.path.exists(src_in_path) and not os.path.exists(tgt_in_path):
            continue
        assert os.path.exists(src_in_path) and os.path.exists(tgt_in_path)

        src_out_path = os.path.join(output_dir, src_filename)
        tgt_out_path = os.path.join(output_dir, tgt_filename)

        if subset == 'test':
            logger.warning(f"Be careful! Filtering some instances from test!")

        filter(src_in_path=src_in_path,
               tgt_in_path=tgt_in_path,
               src_out_path=src_out_path,
               tgt_out_path=tgt_out_path,
               qe=qe)


class ModelFrontQE(QE):

    def __init__(self,
                 src: str,
                 tgt: str,
                 threshold: int,
                 token: str = None):
        self.src = src
        self.tgt = tgt
        self.threshold = threshold / 100
        self.token = token
        self.url = f"https://api.modelfront.com/v1/predict?sl={src}&tl={tgt}&token={token}"
        self.session = None

    def __enter__(self):
        self.session = requests.Session()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.session.close()

    def check(self, src_line: str, tgt_line: str) -> bool:
        return self.check_batch([src_line], [tgt_line])[0]

    def check_batch(self,
                    src_lines: List[str],
                    tgt_lines: List[str]) -> List[bool]:
        rows = []
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            row = {
                "original": src_line,
                "translation": tgt_line
            }
            rows.append(row)

        request_json = {
            "rows": rows
        }

        if self.session is None:
            raise Exception('Session is not open. Please use `with ModelFrontQE(...) as qe:`')
        response = self.session.post(self.url, json=request_json)

        if response.status_code != 200:
            logger.warning(f"Got response {response.status_code}")
            logger.warning(f"Waiting for 10 seconds to retry...")
            time.sleep(10)
            logger.warning(f"Trying again...")
            return self.check_batch(src_lines, tgt_lines)

        response_json = response.json()['rows']

        return [
            row['risk'] <= self.threshold
            for row in response_json
        ]


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('input_dir', type=str, default='')

    parser.add_argument('--qe', type=str, default=None)
    parser.add_argument('--qe-dir', type=str, default=None)
    parser.add_argument('--qe-thresold', type=str, default=50)

    parser.add_argument('--qe-token', type=str, default="None")

    parser.add_argument('--prep-dir', type=str, default=None)
    parser.add_argument('--vocab', default='auto', type=str)

    parser.add_argument('--clean-dir', type=str, default=None)
    parser.add_argument('--no-clean', action='store_true')
    parser.add_argument('--ratio', type=float, default=1.5)
    parser.add_argument('-min', '--min-len', type=int, default=1)
    parser.add_argument('-max', '--max-len', type=int, default=250)

    parser.add_argument('--bin-dir', type=str, default=None)
    parser.add_argument('-j', '--num-workers', type=int, default=1)
    parser.add_argument('--num-bin-shards', type=int, default=1)

    parser.add_argument('--broken-up-sacremoses-cli', action='store_true')


    args = parser.parse_args()

    if not args.qe:
        assert args.qe_dir is None
        args.qe_dir = args.input_dir
    if args.qe_dir is None:
        args.qe_dir = os.path.join(args.input_dir, args.qe)

    if args.prep_dir is None:
        default_dir_name = 'prep' if args.broken_up_sacremoses_cli else 'FSQ'
        args.prep_dir = os.path.join(args.qe_dir, default_dir_name)

    if args.no_clean:
        assert args.clean_dir is None
        args.clean_dir = args.prep_dir
    if args.clean_dir is None:
        args.clean_dir = os.path.join(args.prep_dir, 'clean')

    if args.bin_dir is None:
        args.bin_dir = os.path.join(args.clean_dir, 'bin')

    src, tgt = detect_langs(args.input_dir)

    if args.vocab == 'auto':
        vocab_model = PREDEFINED_VOCAB_MODELS[src, tgt]
    else:
        vocab_model = args.vocab

    logger.warning(f"Using vocab model {vocab_model}")

    if args.qe:
        if args.qe == 'mf':
            qe = ModelFrontQE(src=src,
                              tgt=tgt,
                              threshold=args.qe_thresold,
                              token=args.qe_token)
        else:
            raise NotImplementedError

        with qe:
            filter_dir(input_dir=args.input_dir,
                       output_dir=args.qe_dir,
                       src=src, tgt=tgt,
                       qe=qe)

    prep_dir(input_dir=args.qe_dir,
             output_dir=args.prep_dir,
             src=src, tgt=tgt,
             vocab_model=vocab_model,
             num_workers=args.num_workers)

    if not args.no_clean:
        clean_dir(input_dir=args.prep_dir,
                  output_dir=args.clean_dir,
                  src=src, tgt=tgt,
                  ratio=args.ratio,
                  min_len=args.min_len,
                  max_len=args.max_len)

    binarize_dir(input_dir=args.clean_dir,
                 output_dir=args.bin_dir,
                 src=src, tgt=tgt,
                 vocab_model=vocab_model,
                 num_workers=args.num_bin_shards)


if __name__ == "__main__":
    main()
