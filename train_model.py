import argparse
import logging
import os
import torch
import datasets
from preprocess_timit import process_timit
from preprocess_librispeech import process_librispeech
from pathlib import Path


def train_model(
    dataset: str,
    feat_type: str = "fbank",
    data_format: str = "numpy",
    model_type: str = "fhvae",
    alpha_dis: float = 10.0,
    n_epochs: int = 100,
    n_patience: int = 10,
    steps_per_epoch: int = 5000,
    print_steps: int = 200,
):
    """Loads data and trains the model

    Args:
        dataset:         The dataset to use for training
        feat_type:       Type of feature to compute (only affects numpy data format)
        data_format:     How the computed features should be stored
        model_type:      Type of model to use for training
        alpha_dis:       Discriminative objective weight
        n_epochs:        Maximum number of training epochs
        n_patience:      Maximum number of consecutive epochs without improvement
        steps_per_epoch: Training steps per epoch
        print_steps:     Interval for printing statistics

    """

    # load data
    root_dir = Path(f"./datasets/{args.dataset}")
    if dataset == "timit":
        process_timit(str(root_dir), feat_type, data_format)
    else:
        process_librispeech(str(root_dir), feat_type, data_format)
    # if args.is_numpy:
    #   prepare_numpy(args.dataset, ...)
    # else:
    #   prepare_kaldi(args.dataset, ...)

    # load model

    # set up experiment directory

    # run training


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
    parser.add_argument(
        "--dataset", type=str, default="librispeech", help="Dataset to use"
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
    args = parser.parse_args()
    print(args)

    train_model(
        args.dataset,
        args.ftype,
        args.data_format,
        args.model,
        args.alpha_dis,
        args.n_epochs,
        args.n_patience,
        args.n_steps_per_epoch,
        args.n_print_steps,
    )
