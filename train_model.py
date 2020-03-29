import sys
import argparse
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
from typing import Iterable, Optional
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
    "--feat_type",
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
    "--model-type",
    type=str,
    default="fhvae",
    choices=["fhvae", "simple_fhvae"],
    help="Model architecture",
)
parser.add_argument(
    "--alpha-dis", type=float, default=10.0, help="Discriminative objective weight"
)
parser.add_argument(
    "--epochs", type=int, default=100, help="Number of maximum training epochs"
)
parser.add_argument(
    "--patience",
    type=int,
    default=10,
    help="Number of maximum consecutive non-improving epochs",
)
parser.add_argument(
    "--steps-per-epoch",
    type=int,
    default=5000,
    help="Number of training steps per epoch",
)
parser.add_argument(
    "--log-interval", type=int, default=200, help="Step interval for printing info",
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=200,
    help="Number of steps to save checkpoints",
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
    "--tensorboard",
    action="store_true",
    dest="tensorboard",
    help="Enable Tensorboard logging",
)
parser.add_argument(
    "--visdom", action="store_true", dest="visdom", help="Enable Visdom logging"
)
parser.add_argument(
    "--log-dir", default="visualize/tensorboard", help="Location of tensorboard log",
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
parser.add_argument(
    "--finetune",
    dest="finetune",
    action="store_true",
    help="Fine-tune the model from the provided checkpoint",
)
args = parser.parse_args()
print(args)


# alpha/discriminative weight of 10 was found to produce best results
def loss_function(lower_bound, log_qy, alpha=10.0):
    """Discriminative segment variational lower bound

    Returns:
        Segment variational lower bound plus the (weighted) discriminative objective.

    """

    return -1 * torch.mean(lower_bound + alpha * log_qy)


def save_ckp(
    model,
    optimizer,
    args,
    summary_list,
    values_dict,
    run_info: str,
    epoch: int,
    iteration: Optional[int],
    val_lower_bound: float,
    best_val_lb: float,
    checkpoint_dir: str,
    best_model_path: str,
):
    """Saves checkpoint files"""
    if val_lower_bound > best_val_lb:
        is_best = True

    checkpoint = {
        "args": args,
        "best_val_lb": best_val_lb,
        "epoch": epoch,
        "iteration": iteration,
        "model_params": (
            model.z1_hus,
            model.z2_hus,
            model.z1_dim,
            model.z2_dim,
            model.x_hus,
        ),
        "optimizer": optimizer.state_dict(),
        "state_dict": model.state_dict(),
        "summary_vals": summary_list,
        "values": values_dict,
    }

    f_path = Path(checkpoint_dir) / f"{model}_{run_info}_e{epoch}_i{iteration}.tar"
    torch.save(checkpoint, f_path)
    if is_best:
        shutil.copyfile(f_path, Path(best_model_path))


def check_terminate(epoch, best_epoch, patience, epochs):
    """Checks if training should be terminated"""

    if (epoch - 1) - best_epoch > patience:
        return True
    if epoch > epochs:
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

train_loss_results, val_loss_results, lower_bound_results, discrim_loss_results = (
    torch.Tensor(args.epochs),
    torch.Tensor(args.epochs),
    torch.Tensor(args.epochs),
    torch.Tensor(args.epochs),
)

base_string = f"{args.dataset}_{args.data_format}_{args.feat_type}"
exp_string = f"{args.model_type}_e{args.epochs}_s{args.steps_per_epoch}_p{args.patience}_a{args.alpha_dis}"
run_id = f"{base_string}_{exp_string}"

os.makedirs(args.checkpoint_dir, exist_ok=True)

if args.visdom:
    visdom_logger = VisdomLogger(run_id, args.epochs)
if args.tensorboard:
    tensorboard_logger = TensorBoardLogger(run_id, args.log_dir, args.log_params)

best_epoch, stert_epoch, best_val_lb = 0, 0, -np.inf
optim_state = None
# Load a previously saved checkpoint
if args.continue_from:
    print(f"Loading {args.continue_from}.")
    strict_mode = True
    checkpoint = torch.load(args.continue_from)
    model_type = checkpoint["args"].model_type
    model_params = checkpoint["model_params"]
    if model_type == "fhvae":
        model = FHVAE(*model_params)
    elif model_type == "simple_fhvae":
        model = SimpleFHVAE(*model_params)
    else:
        # Fallback just in case
        model = args.model_type
        strict_mode = False
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    # We don't want to restart training if this is the case
    if not args.finetune:
        optim_state = checkpoint["optimizer"]
        start_epoch = checkpoint["epoch"]
        best_val_lb = checkpoint["best_val_lb"]
        start_iter = checkpoint.get("iteration", None)
        summary_list = checkpoint["summary_vals"]
        if start_iter is None:
            values = checkpoint["values"]
            train_loss = values["train_loss_results"]
            val_loss = values["val_loss_results"]
            lower_bound_results = values["lower_bound_results"]
            discrim_loss_results = values["discrim_loss_results"]
            # Checkpoint was saved at the end of an epoch, start from next epoch
            start_epoch += 1
            start_iter = 0
            attrs = [
                "feat_type",
                "data_format",
                "alpha_dis",
                "epochs",
                "patience",
                "steps_per_epoch",
                "log_interval",
                "checkpoint_interval",
                "training_batch_size",
                "dev_batch_size",
                "tensorboard",
                "visdom",
                "log_dir",
                "log_params",
                "checkpoint_dir",
                "best_model_dir",
            ]
            for attr in attrs:
                vars(args)[attr] = vars(checkpoint["args"])[attr]
            base_string = f"{args.dataset}_{args.data_format}_{args.feat_type}"
            exp_string = f"{args.model_type}_e{args.epochs}_s{args.steps_per_epoch}_p{args.patience}_a{args.alpha_dis}"
            run_id = f"{base_string}_{exp_string}"
            if args.visdom:
                visdom_logger = VisdomLogger(run_id, args.epochs)
                visdom_logger.load_previous_values(start_epoch, values)
            if args.tensorboard:
                tensorboard_logger = TensorBoardLogger(
                    run_id, args.log_dir, args.log_params
                )
                tensorboard_logger.load_previous_values(start_epoch, values)
else:
    data_sets = ("train", "dev", "test")
    # load data
    dataset_directory = create_output_dir(
        args.dataset, args.feat_type, args.data_format
    )

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
                    args.feature_scp,
                    args.length_scp,
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
                    args.feat_ark,
                    args.feature_scp,
                    args.length_scp,
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
        model = SimpleFHVAE(
            args.z1_hus, args.z2_hus, args.z1_dim, args.z2_dim, args.x_hus
        )

    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, betas=(args.beta_one, args.beta_two)
    )

model.to(device)
if optim_state is not None:
    optimizer.load_state_dict(optim_state)

# set up experiment directory
exp_root = Path("./experiments") / base_string
exp_dir = exp_root / exp_string
os.makedirs(exp_dir, exist_ok=True)

for epoch in range(start_epoch, args.epochs):
    # training
    model.train()
    train_loss = 0.0
    for batch_idx, data in enumerate(train_loader, start=start_iter):
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
    summary_list = [0, 0, 0, 0, 0]
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
            if idx + 1 % args.checkpoint_interval == 0:
                save_ckp(
                    model,
                    optimizer,
                    args,
                    summary_list,
                    None,
                    base_string,
                    epoch,
                    idx,
                    val_lower_bound,
                    best_val_lb,
                    args.checkpoint_dir,
                    args.model_dir,
                )

    val_loss /= len(val_loader.dataset)
    print(f"====> Validation set loss: {val_loss:.4f}")

    train_loss_results[epoch] = train_loss
    val_loss_results[epoch] = val_loss
    lower_bound_results[epoch] = val_lower_bound
    discrim_loss_results[epoch] = discrim_loss
    values = {
        "train_loss_results": train_loss_results,
        "val_loss_results": val_loss_results,
        "lower_bound_results": lower_bound_results,
        "discrim_loss_results": discrim_loss_results,
    }
    if args.tensorboard:
        tensorboard_logger.update(epoch, values, model.named_parameters())
    if args.visdom:
        visdom_logger.update(epoch, values)

    # Iteration is None here so we know this was saved at the end of an epoch
    save_ckp(
        model,
        optimizer,
        args,
        summary_list,
        values,
        base_string,
        epoch,
        None,
        val_lower_bound,
        best_val_lb,
        args.checkpoint_dir,
        args.model_dir,
    )

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}"
    )

    if check_terminate(epoch, best_epoch, args.patience, args.epochs):
        print("Training terminated!")
        break
