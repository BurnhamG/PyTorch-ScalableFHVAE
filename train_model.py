import sys
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
from logger import VisdomLogger, TensorBoardLogger
import shutil
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, help="Dataset to use")
parser.add_argument(
    "--raw-data-dir", type=str, default=None, help="Location of the raw data"
)
parser.add_argument(
    "--ftype",
    type=str,
    default="fbank",
    choices=["fbank", "spec"],
    help="Feature type to compute (only affects numpy data)",
)
parser.add_argument(
    "--data-format",
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
    "--alpha-dis", type=float, default=10.0, help="Discriminative objective weight"
)
parser.add_argument(
    "--n-epochs", type=int, default=100, help="Number of maximum training epochs"
)
parser.add_argument(
    "--n-patience",
    type=int,
    default=10,
    help="Number of maximum consecutive non-improving epochs",
)
parser.add_argument(
    "--n-steps-per-epoch",
    type=int,
    default=5000,
    help="Number of training steps per epoch",
)
parser.add_argument(
    "--n-print-steps",
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
    "--learning-rate", type=float, default=0.001, help="Learning rate for training"
)
parser.add_argument(
    "--beta-one", type=float, default=0.95, help="Beta1 for the Adam optimizer"
)
parser.add_argument(
    "--beta-two", type=float, default=0.999, help="Beta2 for the Adam optimizer"
)
parser.add_argument(
    "--feature-scp", type=str, default=None, help="Path to the feature scp file"
)
parser.add_argument(
    "--length-scp", type=str, default=None, help="Path to the length scp file"
)
parser.add_argument(
    "--sample-rate",
    type=int,
    default=None,
    help="Sample rate to use for resampling audio samples",
)
parser.add_argument(
    "--win-size",
    type=float,
    default=0.025,
    help="Window size for spectrogram in seconds",
)
parser.add_argument(
    "--hop-size",
    type=float,
    default=0.010,
    help="Window stride for spectrogram in seconds",
)
parser.add_argument("--n-mels", type=int, default=80, help="Number of filter banks")
parser.add_argument(
    "--feat-ark",
    type=str,
    default=None,
    help="Path to the preprocessed Kaldi .ark file",
)
parser.add_argument(
    "--fbank-conf",
    type=str,
    default=None,
    help="Path to the fbank.conf file kaldi should use",
)
parser.add_argument(
    "--kaldi-root", type=str, default=None, help="Root directory for Kaldi"
)
parser.add_argument(
    "--min-len",
    type=int,
    default=None,
    help="Minimum segment length. If none is provided this will use the sequence length",
)
parser.add_argument(
    "--mvn-path",
    type=str,
    default=None,
    help="Path to a precomputed mean and variance normalization file",
)
parser.add_argument("--seg-len", type=int, default=20, help="Segment length to use")
parser.add_argument(
    "--seg-shift",
    type=int,
    default=8,
    help="Segment shift if rand_seg is False, otherwise floor(seq_len/seg_shift) segments per sequence will be extracted",
)
parser.add_argument(
    "--rand-seg",
    type=bool,
    default=False,
    help="If True, segments will be randomly extracted",
)
parser.add_argument(
    "--training-batch-size",
    type=int,
    default=256,
    help="Batch size to use for training",
)
parser.add_argument(
    "--dev-batch-size",
    type=int,
    default=256,
    help="Batch size to use for evaluation against the development set",
)
parser.add_argument(
    "--z1-hus",
    default=[128, 128],
    nargs=2,
    help="List of the number of hideen units for the two layers of z1",
)
parser.add_argument(
    "--z2-hus",
    default=[128, 128],
    nargs=2,
    help="List of the number of hideen units for the two layers of z2",
)
parser.add_argument(
    "--z1-dim", type=int, default=16, help="Dimensionality of the z1 layer"
)
parser.add_argument(
    "--z2-dim", type=int, default=16, help="Dimensionality of the z2 layer"
)
parser.add_argument(
    "--x-hus",
    default=[128, 128],
    nargs=2,
    help="List of hidden units per layer for the pre-stochastic layer decoder",
)
parser.add_argument(
    "--device", type=str, default="gpu", help="Device to use for computations"
)
parser.add_argument(
    "--log-interval",
    type=int,
    help="Step interval for printing information and saving checkpoints",
)
parser.add_argument(
    "--tensorboard",
    action="store_true",
    dest="tensorboard",
    help="Enable Tensorboard logging",
)
parser.add_argument(
    "--visdom", action="store_true", dest="visdom", help="Enable Visdom logging"
)
parser.add_argument(
    "--log-dir",
    default="visualize/deepspeech_final",
    help="Location of tensorboard log",
)
parser.add_argument(
    "--log-params",
    dest="log_params",
    action="store_true",
    help="Log parameter values and gradients",
)
parser.add_argument(
    "--checkpoint-dir",
    default="./models",
    type=str,
    help="Directory that will hold the model checkpoints",
)
parser.add_argument(
    "--continue-from",
    default="",
    type=str,
    help="Checkpoint model for continuing training",
)
parser.add_argument(
    "--best-model-dir",
    default="./models/best_model.pth",
    help="Location to save the best epoch models",
)
args = parser.parse_args()
print(args)


# alpha/discriminative weight of 10 was found to produce best results
def loss_function(lower_bound, log_qy, alpha=10.0):
    """Discriminative segment variational lower bound

    Segment variational lower bound plus the (weighted) discriminative objective

    """

    return -1 * torch.mean(lower_bound + alpha * log_qy)


def save_ckp(
    state,
    model,
    run_info: str,
    epoch,
    iteration,
    is_best,
    checkpoint_dir,
    best_model_path,
):
    f_path = Path(checkpoint_dir) / f"{model}_{run_info}_e{epoch}_i{iteration}.tar"
    torch.save(state, f_path)
    if is_best:
        shutil.copyfile(f_path, Path(best_model_path))


def check_terminate(epoch, best_epoch, n_patience, n_epochs):
    if (epoch - 1) - best_epoch > n_patience:
        return True
    if epoch > n_epochs:
        return True
    return False


if args.raw_data_dir is None and args.is_preprocessed is False:
    raise ValueError(
        "You must provide a raw data location if the data is not preprocessed!"
    )

if args.min_len is None:
    min_len = args.seg_len

if args.device == "gpu":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

loss_results, lower_bound_results, discrim_loss_results = (
    torch.Tensor(args.n_epochs),
    torch.Tensor(args.n_epochs),
    torch.Tensor(args.n_epochs),
)

base_string = f"{args.dataset}_{args.data_format}_{args.feat_type}"
exp_string = f"{args.model_type}_e{args.n_epochs}_s{args.steps_per_epoch}_p{args.n_patience}_a{args.alpha_dis}"
run_id = f"{base_string}_{exp_string}"

os.makedirs(args.checkpoint_dir, exist_ok=True)

if args.visdom:
    visdom_logger = VisdomLogger(run_id, args.n_epochs)
if args.tensorboard:
    tensorboard_logger = TensorBoardLogger(run_id, args.log_dir, args.log_params)

if args.continue_from:
    print(f"Loading {args.continue_from}.")
    strict_mode = True
    package = torch.load(args.continue_from)
    model = package["args"].get("model_type")
    if model is None:
        model = args.model_type
        strict_mode = False
    model.load_state_dict(package["state_dict"], strict=False)
    # Fallback just in case
    if not args.finetune:
        optim_state = package["optimizer"]
        start_epoch = package["epoch"] + 1  # Saved epoch is last full epoch
        values = package["values"]
        val_loss = package["val_loss"]
        lower_bound_results = values["lower_bound_results"]
        discrim_loss_results = values["discrim_loss_results"]
        best_val_lb = lower_bound_results[start_epoch]


# Handle any extraneous arguments that may have been passed
args.z1_hus = args.z1_hus[:2]
args.z2_hus = args.z2_hus[:2]
args.x_hus = args.x_hus[:2]

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
                args.win_size,
                args.hop_size,
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

train_dataset_args = [
    paths_dict["train"]["feat_pth"],
    paths_dict["train"]["len_pth"],
    min_len,
    args.mvn_path,
    args.seg_len,
    args.seg_shift,
    args.rand_seg,
]
dev_dataset_args = [
    paths_dict["dev"]["feat_pth"],
    paths_dict["dev"]["len_pth"],
    min_len,
    args.mvn_path,
    args.seg_len,
    args.seg_shift,
    args.rand_seg,
]

if args.data_format == "numpy":
    train_dataset = NumpyDataset(*train_dataset_args)
    dev_dataset = NumpyDataset(*dev_dataset_args)
else:
    train_dataset = KaldiDataset(*train_dataset_args)
    dev_dataset = KaldiDataset(*dev_dataset_args)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.training_batch_size, shuffle=True, num_workers=4
)
val_loader = torch.utils.data.DataLoader(
    dev_dataset, batch_size=args.dev_batch_size, shuffle=True, num_workers=4
)

# load model
if args.model_type == "fhvae":
    model = FHVAE(args.z1_hus, args.z2_hus, args.z1_dim, args.z2_dim, args.x_hus)
else:
    model = SimpleFHVAE(args.z1_hus, args.z2_hus, args.z1_dim, args.z2_dim, args.x_hus)

model.to(device)
# set up experiment directory
exp_root = Path("./experiments") / base_string
exp_dir = exp_root / exp_string
os.makedirs(exp_dir, exist_ok=True)

# run training
optimizer = Adam(
    model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2)
)

best_epoch, best_val_lb = 0, -np.inf
for epoch in range(args.n_epochs):
    # training
    model.train()
    train_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        # tr_summ_vars are log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2
        lower_bound, discrim_loss, tr_summ_vars = model(*data, len(train_dataset))
        loss = loss_function(lower_bound, discrim_loss, args.alpha_dis)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx + 1 % args.log_interval == 0:
            current_pos = batch_idx * len(data)
            tot_len = len(train_loader.dataset)
            percentage = 100.0 * batch_idx / len(train_loader)
            cur_loss = loss.item() / len(data)

            print(
                f"====> Train Epoch: {epoch} [{current_pos}/{tot_len} ({percentage:.0f}%)]\tLoss: {cur_loss:.6f}"
            )
            if np.isnan(lower_bound):
                print("Training diverged")
                raise sys.exit(2)

    train_loss /= len(val_loader.dataset)
    print(f"====> Train set average loss: {train_loss:.4f}")

    # eval
    model.eval()
    val_loss = 0.0
    summary_list = [0, 0, 0, 0, 0, 0]
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            data = data.to(device)
            val_lower_bound, _, val_summ_vars = model(*data, len(val_loader.dataset))
            val_loss += loss_function(lower_bound, discrim_loss, args.alpha_dis).item()
            summary_list = [
                map(torch.sum, summary_list, (val_lower_bound, *val_summ_vars))
            ]
            if idx + 1 % args.log_interval == 0:
                current_pos = batch_idx * len(data)
                tot_len = len(train_loader.dataset)
                percentage = 100.0 * batch_idx / len(train_loader)
                cur_loss = loss.item() / len(data)

                print(
                    f"====> Validation Epoch: {epoch} [{current_pos}/{tot_len} ({percentage:.0f}%)]\tLoss: {cur_loss:.6f}"
                )

    val_loss /= len(val_loader.dataset)
    print(f"====> Validation set loss: {val_loss:.4f}")

    lower_bound_results[epoch] = val_lower_bound
    values = {
        "lower_bound_results": lower_bound_results,
        "discrim_loss_results": discrim_loss_results,
    }
    if args.tensorboard:
        tensorboard_logger.update(epoch, values, model.named_parameters())
    if args.visdom:
        visdom_logger.update(epoch, values)

    if val_lower_bound > best_val_lb:
        is_best = True

    # Save model
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": args,
        "best_val_lb": best_val_lb,
    }

    save_ckp(
        checkpoint,
        args.model_type,
        base_string,
        epoch,
        len(val_loader.dataset),
        is_best,
        args.checkpoint_dir,
        args.model_dir,
    )

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}"
    )

    if check_terminate(epoch, best_epoch, args.n_patience, args.n_epochs):
        print("Training terminated!")
        break
