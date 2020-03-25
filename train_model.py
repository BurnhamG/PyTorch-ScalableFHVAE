import argparse
import logging
import os
import torch
import datasets
from preprocess_timit import process_timit
from preprocess_librispeech import process_librispeech
from pathlib import Path
from prepare_numpy_data import prepare_numpy
from prepare_kaldi_data import prepare_kaldi
import time
from multiprocessing import Pool
from utils import create_output_dir
from simple_fhvae import SimpleFHVAE
from fhvae import FHVAE
from torch.optim import Adam
from typing import Iterable
from datasets import NumpyDataset, KaldiDataset


def train_model(
    dataset: str,
    raw_data_dir: str,
    feat_type: str = "fbank",
    data_format: str = "numpy",
    model_type: str = "fhvae",
    alpha_dis: float = 10.0,
    n_epochs: int = 100,
    n_patience: int = 10,
    steps_per_epoch: int = 5000,
    print_steps: int = 200,
    is_preprocessed: bool = True,
    learning_rate: float = 0.001,
    beta1: float = 0.95,
    beta2: float = 0.999,
):
    """Loads data and trains the model

    Args:
        dataset:         The dataset to use for training
        raw_data_dir:
        feat_type:       Type of feature to compute (only affects numpy data format)
        data_format:     How the computed features should be stored
        model_type:      Type of model to use for training
        alpha_dis:       Discriminative objective weight
        n_epochs:        Maximum number of training epochs
        n_patience:      Maximum number of consecutive epochs without improvement
        steps_per_epoch: Training steps per epoch
        print_steps:     Interval for printing statistics
        is_preprocessed:

    """
    if args.raw_data_dir is None and args.is_preprocessed is False:
        raise ValueError(
            "You must provide a raw data location if the data is not preprocessed!"
        )

    data_sets = ("train", "dev", "test")
    # load data
    dataset_directory = create_output_dir(dataset, feat_type, data_format)

    # paths is (training_wav_scp, dev_wav_scp, test_wav_scp)
    if not is_preprocessed:
        if dataset == "timit":
            paths = process_timit(Path(raw_data_dir), dataset_directory)
        else:
            paths = process_librispeech(Path(raw_data_dir), dataset_directory)

        starmap_args = []
        if args.is_numpy:
            for set_name, wav_scp in zip(data_sets, paths):
                func_args = [
                    dataset,
                    set_name,
                    wav_scp,
                    dataset_directory,
                    feat_scp,
                    len_scp,
                    feat_type,
                    sample_rate,
                    win_t,
                    hop_t,
                    n_mels,
                ]
                starmap_args.append(tuple(func_args))
            files_start_time = time.time()
            with Pool(3) as p:
                results: Iterable = p.starmap(prepare_numpy, starmap_args)

            # results is a list of tuples of (files_processed, (wav_pth, feat_pth, len_pth))
            tot_files = sum(r[0] for r in results)
            paths_dict = {
                ds: {
                    name: p for name, p in zip(("wav_pth", "feat_pth", "len_pth"), pth)
                }
                for ds, pth in zip(data_sets, [r[1] for r in results])
            }

            print(
                f"Processed {tot_files} files in {time.time() - files_start_time} seconds."
            )
        else:
            for wav_scp in paths[1:]:
                func_args = [
                    str(wav_scp),
                    feat_ark,
                    feat_scp,
                    len_scp,
                    fbank_conf,
                    kaldi_root,
                ]
                starmap_args.append(tuple(func_args))
            with Pool(3) as p:
                results = p.starmap(prepare_kaldi, starmap_args)
            print("Processing complete")

    dataset_args = [
        paths_dict["train"]["feat_pth"],
        paths_dict["train"]["len_pth"],
        min_len,
        mvn_path,
        seg_len,
        seg_shift,
        rand_seg,
    ]

    if data_format == "numpy":
        audio_dataset = NumpyDataset(*dataset_args)
    else:
        audio_dataset = KaldiDataset(*dataset_args)
    audio_loader = torch.utils.data.DataLoader(
        audio_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    # load model
    if model_type == "fhvae":
        model = FHVAE()
    else:
        model = SimpleFHVAE(z1_hus, z2_hus, z1_dim, z2_dim, x_hus)
    # set up experiment directory
    exp_root = Path("./experiments") / dataset
    exp_dir = (
        exp_root
        / f"{model_type}_e{n_epochs}_s{steps_per_epoch}_p{n_patience}_a{alpha_dis}"
    )
    os.makedirs(exp_dir, exist_ok=True)

    # run training
    optimizer = Adam(lr=learning_rate, betas=(beta1, beta2))


# alpha/discriminative weight of 10 was found to produce best results
def loss_function(lower_bound, log_qy, alpha=10.0):
    """
    Discriminative segment variational lower bound
    Segment variational lower bound plus the (weighted) discriminative objective
    """

    return -1 * torch.mean(lower_bound + alpha * log_qy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument(
        "--raw_data_dir", type=str, default=None, help="Location of the raw data"
    )
    parser.add_argument(
        "--ftype",
        type=str,
        default="fbank",
        choices=["fbank", "spec"],
        help="Feature type to compute (only affects numpy data)",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="numpy",
        choices=["kaldi", "numpy"],
        help="Format used to store data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fhvae",
        help="Model architecture; {fhvae|simple_fhvae}",
    )
    parser.add_argument(
        "--alpha_dis", type=float, default=10.0, help="Discriminative objective weight"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="Number of maximum training epochs"
    )
    parser.add_argument(
        "--n_patience",
        type=int,
        default=10,
        help="Number of maximum consecutive non-improving epochs",
    )
    parser.add_argument(
        "--n_steps_per_epoch",
        type=int,
        default=5000,
        help="Number of training steps per epoch",
    )
    parser.add_argument(
        "--n_print_steps",
        type=int,
        default=200,
        help="Number of steps to print statistics",
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        dest="is_preprocessed",
        help="Use this flag if the data is already preprocessed",
    )
    args = parser.parse_args()
    print(args)

    train_model(
        args.dataset,
        args.raw_data_dir,
        args.ftype,
        args.data_format,
        args.model,
        args.alpha_dis,
        args.n_epochs,
        args.n_patience,
        args.n_steps_per_epoch,
        args.n_print_steps,
        args.is_preprocessed,
    )
