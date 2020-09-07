import torch
from utils import create_output_dir_name
from preprocess_timit import process_timit
from preprocess_librispeech import process_librispeech
from pathlib import Path
from prepare_numpy_data import prepare_numpy
from prepare_kaldi_data import prepare_kaldi
import time
from multiprocessing import Pool
from typing import Iterable
import argparse


def preprocess_data(args):
    data_sets = ("train", "dev", "test")
    # load data
    dataset_directory = create_output_dir_name(
        args.dataset, args.data_format, args.feat_type
    )

    # paths is (training_wav_scp, dev_wav_scp, test_wav_scp)
    if args.dataset == "timit":
        process_timit(Path(args.raw_data_dir).resolve(), dataset_directory)
    else:
        process_librispeech(
            Path(args.raw_data_dir).resolve(), dataset_directory, args.data_format
        )

    starmap_args = []
    if args.data_format == "numpy":
        for set_name in data_sets:
            func_args = [
                args.dataset,
                set_name,
                dataset_directory,
                None,
                args.feat_type,
                args.sample_rate,
                args.win_size,
                args.hop_size,
                args.mels,
            ]
            starmap_args.append(tuple(func_args))
        files_start_time = time.time()
        with Pool(3) as p:
            results: Iterable = p.starmap(prepare_numpy, starmap_args)
    else:
        for set_name in data_sets:
            func_args = [dataset_directory, set_name, args.fbank_conf, args.kaldi_root]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["librispeech", "timit"],
        help="Dataset to preprocess",
    )
    parser.add_argument("raw_data_dir", type=str, help="Location for raw data")
    parser.add_argument(
        "--data-format",
        type=str,
        default="numpy",
        choices=["numpy", "kaldi"],
        help="Data format to use for precomputed features",
    )
    parser.add_argument(
        "--fbank-conf",
        type=str,
        default="./misc/fbank.conf",
        help="Kaldi fbank configuration",
    )
    parser.add_argument(
        "--feat-type",
        type=str,
        default="fbank",
        choices=["fbank", "spec"],
        help="Feature type to compute (only affects numpy data)",
    )
    parser.add_argument(
        "--hop-size", type=float, default=0.010, help="Frame spacing in seconds"
    )
    parser.add_argument(
        "--kaldi-root", type=str, default="./kaldi", help="Kaldi root directory"
    )
    parser.add_argument(
        "--mels", type=int, default=80, help="Number of filter banks if choosing fbank",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Resample raw audio to specified value if not None",
    )
    parser.add_argument(
        "--win-size", type=float, default=0.025, help="Window size in seconds"
    )
    args = parser.parse_args()

    preprocess_data(args)
