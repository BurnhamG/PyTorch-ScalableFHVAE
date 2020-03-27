import argparse
import logging
import os
import torch
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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
parser.add_argument(
    "--learning_rate", type=float, default=0.001, help="Learning rate for training"
)
parser.add_argument(
    "--beta_one", type=float, default=0.95, help="Beta1 for the Adam optimizer"
)
parser.add_argument(
    "--beta_two", type=float, default=0.999, help="Beta2 for the Adam optimizer"
)
parser.add_argument(
    "--feature_scp", type=str, default=None, help="Path to the feature scp file"
)
parser.add_argument(
    "--length_scp", type=str, default=None, help="Path to the length scp file"
)
parser.add_argument(
    "--sample_rate",
    type=int,
    default=None,
    help="Sample rate to use for resampling audio samples",
)
parser.add_argument("--win_t", type=float, default=0.025)
parser.add_argument("--hop_t", type=float, default=0.010)
parser.add_argument("--n_mels", type=int, default=80)
parser.add_argument(
    "--feat_ark",
    type=str,
    default=None,
    help="Path to the preprocessed Kaldi .ark file",
)
parser.add_argument(
    "--fbank_conf",
    type=str,
    default=None,
    help="Path to the fbank.conf file kaldi should use",
)
parser.add_argument(
    "--kaldi_root", type=str, default=None, help="Root directory for Kaldi"
)
parser.add_argument(
    "--min_len",
    type=int,
    default=None,
    help="Minimum segment length. If none is provided this will use the sequence length",
)
parser.add_argument(
    "--mvn_path",
    type=str,
    default=None,
    help="Path to a precomputed mean and variance normalization file",
)
parser.add_argument("--seg_len", type=int, default=20, help="Segment length to use")
parser.add_argument(
    "--seg_shift",
    type=int,
    default=8,
    help="Segment shift if rand_seg is False, otherwise floor(seq_len/seg_shift) segments per sequence will be extracted",
)
parser.add_argument(
    "--rand_seg",
    type=bool,
    default=False,
    help="If True, segments will be randomly extracted",
)
parser.add_argument(
    "--training_batch_size",
    type=int,
    default=256,
    help="Batch size to use for training",
)
parser.add_argument(
    "--dev_batch_size",
    type=int,
    default=256,
    help="Batch size to use for evaluation against the development set",
)
parser.add_argument(
    "--z1_hus",
    type=list,
    default=[128, 128],
    help="List of the number of hideen units for the layers of z1",
)
parser.add_argument(
    "--z2_hus",
    type=list,
    default=[128, 128],
    help="List of the number of hideen units for the layers of z2",
)
parser.add_argument(
    "--z1_dim", type=int, default=16, help="Dimensionality of the z1 layer"
)
parser.add_argument(
    "--z2_dim", type=int, default=16, help="Dimensionality of the z2 layer"
)
parser.add_argument(
    "--x_hus",
    type=list,
    default=[128, 128],
    help="List of hidden units per layer for the pre-stochastic layer decoder",
)
parser.add_argument(
    "--device", type=str, default="cuda", help="Device to use for computations"
)
args = parser.parse_args()
print(args)

# def train_model(
#     dataset: str,
#     raw_data_dir: str,
#     feat_type: str = "fbank",
#     data_format: str = "numpy",
#     model_type: str = "fhvae",
#     alpha_dis: float = 10.0,
#     n_epochs: int = 100,
#     n_patience: int = 10,
#     steps_per_epoch: int = 5000,
#     print_steps: int = 200,
#     is_preprocessed: bool = True,
#     learning_rate: float = 0.001,
#     beta1: float = 0.95,
#     beta2: float = 0.999,
#     feat_scp=None,
#     len_scp=None,
#     sample_rate: int = None,
#     win_t=0.025,
#     hop_t=0.010,
#     n_mels=80,
#     feat_ark=None,
#     fbank_conf=None,
#     kaldi_root=None,
#     min_len=None,
#     mvn_path=None,
#     seg_len=20,
#     seg_shift=8,
#     rand_seg=False,
#     batch_size=256,
#     z1_hus=[128, 128],
#     z2_hus=[128, 128],
#     z1_dim=16,
#     z2_dim=16,
#     x_hus=[128, 128],
#     device="cuda",
# ):
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

if args.min_len is None:
    min_len = args.seg_len

data_sets = ("train", "dev", "test")
# load data
dataset_directory = create_output_dir(args.dataset, args.feat_type, args.data_format)

# paths is (training_wav_scp, dev_wav_scp, test_wav_scp)
if not args.is_preprocessed:
    if args.dataset == "timit":
        paths = process_timit(Path(args.raw_data_dir), dataset_directory)
    else:
        paths = process_librispeech(Path(args.raw_data_dir), dataset_directory)

    starmap_args = []
    if args.is_numpy:
        for set_name, wav_scp in zip(data_sets, paths):
            func_args = [
                args.dataset,
                set_name,
                wav_scp,
                dataset_directory,
                args.feat_scp,
                args.len_scp,
                args.feat_type,
                args.sample_rate,
                args.win_t,
                args.hop_t,
                args.n_mels,
            ]
            starmap_args.append(tuple(func_args))
        files_start_time = time.time()
        with Pool(3) as p:
            results: Iterable = p.starmap(prepare_numpy, starmap_args)

        # results is a list of tuples of (files_processed, (wav_pth, feat_pth, len_pth))
        tot_files = sum(r[0] for r in results)
        paths_dict = {
            ds: {name: p for name, p in zip(("wav_pth", "feat_pth", "len_pth"), pth)}
            for ds, pth in zip(data_sets, [r[1] for r in results])
        }

        print(
            f"Processed {tot_files} files in {time.time() - files_start_time} seconds."
        )
    else:
        for wav_scp in paths[1:]:
            func_args = [
                str(wav_scp),
                args.feat_ark,
                args.feat_scp,
                args.len_scp,
                args.fbank_conf,
                args.kaldi_root,
            ]
            starmap_args.append(tuple(func_args))
        with Pool(3) as p:
            results = p.starmap(prepare_kaldi, starmap_args)
        print("Processing complete")

dataset_args = [
    paths_dict["train"]["feat_pth"],
    paths_dict["train"]["len_pth"],
    min_len,
    args.mvn_path,
    args.seg_len,
    args.seg_shift,
    args.rand_seg,
]

if args.data_format == "numpy":
    audio_dataset = NumpyDataset(*dataset_args)
else:
    audio_dataset = KaldiDataset(*dataset_args)
audio_loader = torch.utils.data.DataLoader(
    audio_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)
# load model
if args.model_type == "fhvae":
    model = FHVAE()
else:
    model = SimpleFHVAE(args.z1_hus, args.z2_hus, args.z1_dim, args.z2_dim, args.x_hus)
# set up experiment directory
exp_root = Path("./experiments") / args.dataset
exp_dir = (
    exp_root
    / f"{args.model_type}_e{args.n_epochs}_s{args.steps_per_epoch}_p{args.n_patience}_a{args.alpha_dis}"
)
os.makedirs(exp_dir, exist_ok=True)

# run training
optimizer = Adam(
    model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2)
)

for epoch in range(args.n_epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(audio_loader):
        data = data.to(args.device)
        optimizer.zero_grad()
        lower_bound, discrim_loss = model(*data, len(audio_dataset))
        loss = args.loss_function(lower_bound, discrim_loss, args.alpha_dis)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            current_pos = batch_idx * len(data)
            tot_len = len(audio_loader.dataset)
            percentage = 100.0 * batch_idx / len(audio_loader)
            cur_loss = loss.item() / len(data)
            print(
                f"Train Epoch: {epoch} [{current_pos}/{tot_len} ({percentage:.0f}%)]\tLoss: {cur_loss:.6f}"
            )
    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(audio_loader.dataset):.4f}"
    )


# alpha/discriminative weight of 10 was found to produce best results
def loss_function(lower_bound, log_qy, alpha=10.0):
    """Discriminative segment variational lower bound

    Segment variational lower bound plus the (weighted) discriminative objective

    """

    return -1 * torch.mean(lower_bound + alpha * log_qy)
