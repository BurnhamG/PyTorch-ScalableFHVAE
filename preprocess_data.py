import torch
from utils import create_output_dir
from preprocess_timit import process_timit
from preprocess_librispeech import process_librispeech
from pathlib import Path
from prepare_numpy_data import prepare_numpy
from prepare_kaldi_data import prepare_kaldi
import time
from multiprocessing import Pool
from typing import Iterable


def preprocess_data(args):
    data_sets = ("train", "dev", "test")
    # load data
    dataset_directory = create_output_dir(
        args.dataset, args.feat_type, args.data_format
    )

    # paths is (training_wav_scp, dev_wav_scp, test_wav_scp)
    if args.dataset == "timit":
        paths = process_timit(Path(args.raw_data_dir), dataset_directory)
    else:
        paths = process_librispeech(Path(args.raw_data_dir), dataset_directory)

    starmap_args = []
    if args.data_format == "numpy":
        for set_name, wav_scp in zip(data_sets, paths):
            func_args = [
                args.dataset,
                set_name,
                wav_scp,
                dataset_directory,
                args.feat_type,
                args.sample_rate,
                args.win_size,
                args.hop_size,
                args.n_mels,
            ]
            starmap_args.append(tuple(func_args))
        files_start_time = time.time()
        with Pool(3) as p:
            results: Iterable = p.starmap(prepare_numpy, starmap_args)
    else:
        for wav_scp in paths:
            func_args = [
                str(wav_scp),
                args.fbank_conf,
                args.kaldi_root,
            ]
            starmap_args.append(tuple(func_args))
        with Pool(3) as p:
            results = p.starmap(prepare_kaldi, starmap_args)

    # results is a list of tuples of (files_processed, (wav_pth, feat_pth, len_pth))
    tot_files = sum(r[0] for r in results)
    print(f"Processed {tot_files} files in {time.time() - files_start_time} seconds.")

    file_paths = zip(("wav_pth", "feat_pth", "len_pth"), [r[1] for r in results])
    paths_dict = {ds: {name: pth} for (name, pth) in file_paths for ds in data_sets}

    return paths_dict
