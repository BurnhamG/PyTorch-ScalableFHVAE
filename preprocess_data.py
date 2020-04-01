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
        paths = process_librispeech(
            Path(args.raw_data_dir), dataset_directory, args.data_format
        )

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
        for se, wav_scp in zip(data_sets, paths):
            func_args = [str(wav_scp), args.fbank_conf, args.kaldi_root, se]
            starmap_args.append(tuple(func_args))
        files_start_time = time.time()
        with Pool(3) as p:
            results = p.starmap(prepare_kaldi, starmap_args)

    files_end_time = time.time()
    # results is a list of tuples of (files_processed, (returned file paths))
    tot_files = sum(r[0] for r in results)
    print(
        f"Processed {tot_files} files in {files_end_time - files_start_time:.2f} seconds."
    )

    file_paths = list(zip(data_sets, [r[1] for r in results]))
    if args.data_format == "numpy":
        file_keys = ("wav_pth", "feat_pth", "len_pth")
    else:
        file_keys = ("wav_pth", "feat_ark", "feat_pth", "len_pth")

        paths_dict = {
            se[0]: {file_id: path for (file_id, path) in zip(file_keys, se[1])}
            for se in file_paths
        }

    return paths_dict
