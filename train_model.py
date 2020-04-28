import sys
from utils import create_output_dir
from preprocess_data import preprocess_data
import argparse
import os
import torch
from pathlib import Path
from simple_fhvae import SimpleFHVAE
from fhvae import FHVAE
from torch.optim import Adam
from logger import VisdomLogger, TensorBoardLogger
import numpy as np
from datasets import NumpyDataset, KaldiDataset
from utils import (
    load_checkpoint_file,
    save_checkpoint,
    create_training_strings,
    check_best,
    save_args,
    load_args,
)

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
parser.add_argument("--mels", type=int, default=80, help="Number of filter banks")
parser.add_argument(
    "--fbank-conf",
    type=str,
    default="./misc/fbank.conf",
    help="Path to the fbank.conf file kaldi should use",
)
parser.add_argument(
    "--kaldi-root", type=str, default="./kaldi/", help="Root directory for Kaldi"
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
    default=2048,
    help="Batch size to use for evaluation against the development set",
)
parser.add_argument(
    "--z1-hus",
    default=[128, 128],
    nargs=2,
    help="List of the number of hidden units for the two layers of z1",
)
parser.add_argument(
    "--z2-hus",
    default=[128, 128],
    nargs=2,
    help="List of the number of hidden units for the two layers of z2",
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
    "--tb-log-dir",
    default="./visualize/tensorboard",
    help="Location of tensorboard log",
)
parser.add_argument(
    "--log-params",
    dest="log_params",
    action="store_true",
    help="Log parameter values and gradients",
)
parser.add_argument(
    "--continue-from", type=str, help="Checkpoint model for continuing training",
)
parser.add_argument(
    "--finetune",
    dest="finetune",
    action="store_true",
    help="Fine-tune the model from the provided checkpoint",
)

legacy_opts = parser.add_argument_group(
    "Legacy options",
    "These options can be used to emulate behavior from the original papers more closely.",
)
legacy_opts.add_argument(
    "--legacy",
    dest="legacy",
    action="store_true",
    help="Override 'standard' options with provided legacy options",
)
legacy_opts.add_argument(
    "--steps-per-epoch",
    type=int,
    default=5000,
    help="Number of training steps per epoch",
)
legacy_opts.add_argument(
    "--log-interval", type=int, default=200, help="Step interval for printing info",
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
    args.min_len = args.seg_len

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

start_epoch = 0

base_string, exp_string, run_id = create_training_strings(args)

if args.visdom:
    visdom_logger = VisdomLogger(run_id, args.epochs)
if args.tensorboard:
    tensorboard_logger = TensorBoardLogger(run_id, args.tb_log_dir, args.log_params)

best_epoch, stert_epoch, best_val_lb = 0, 0, -np.inf
optim_state = None
summary_list = None

train_loss_results = {}
val_loss_results = {}
lower_bound_results = {}
discrim_loss_results = {}

# Load a previously saved checkpoint
if args.continue_from:
    print(f"Loading {args.continue_from}.")
    (
        values,
        optim_state,
        start_epoch,
        best_val_lb,
        summary_list,
    ) = load_checkpoint_file(args.continue_from, args.finetune, args.model_type)
    args = load_args(os.path.dirname(args.continue_from))
    base_string, exp_string, run_id = create_training_strings(args)

    # Load previous values into loggers
    if args.visdom:
        visdom_logger = VisdomLogger(run_id, args.epochs)
        visdom_logger.load_previous_values(start_epoch, values)
    if args.tensorboard:
        tensorboard_logger = TensorBoardLogger(run_id, args.tb_log_dir, args.log_params)
        tensorboard_logger.load_previous_values(start_epoch, values)
else:
    # Starting fresh
    if not args.is_preprocessed:
        paths_dict = preprocess_data(args)
        train_dataset_args = [
            paths_dict["train"]["feat_pth"],
            paths_dict["train"]["len_pth"],
            args.min_len,
            args.mvn_path,
            args.seg_len,
            args.seg_shift,
            args.rand_seg,
        ]
        dev_dataset_args = [
            paths_dict["dev"]["feat_pth"],
            paths_dict["dev"]["len_pth"],
            args.min_len,
            args.mvn_path,
            args.seg_len,
            args.seg_shift,
            args.rand_seg,
        ]
    else:
        dataset_dir = create_output_dir(args.dataset, args.data_format, args.feat_type)
        train_dataset_args = [
            dataset_dir / "train" / "feats.scp",
            dataset_dir / "train" / "len.scp",
            args.min_len,
            args.mvn_path,
            args.seg_len,
            args.seg_shift,
            args.rand_seg,
        ]
        dev_dataset_args = [
            dataset_dir / "dev" / "feats.scp",
            dataset_dir / "dev" / "len.scp",
            args.min_len,
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

    if args.legacy:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=1, shuffle=True, num_workers=4
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.training_batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_loader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=args.dev_batch_size, shuffle=True, num_workers=4
        )
    example_data = train_dataset[42][1]

    input_size = np.prod(example_data.shape)
    # load model
    if args.model_type == "fhvae":
        model = FHVAE(
            input_size, args.z1_hus, args.z2_hus, args.z1_dim, args.z2_dim, args.x_hus
        )
    else:
        model = SimpleFHVAE(
            input_size, args.z1_hus, args.z2_hus, args.z1_dim, args.z2_dim, args.x_hus
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

save_args(exp_dir, args)

for epoch in range(start_epoch, args.epochs):
    # training
    model.train()
    train_loss = 0.0
    for batch_idx, (idxs, features, nsegs) in enumerate(train_loader):
        features = features.to(device)
        idxs = torch.tensor([idx for idx in idxs])
        optimizer.zero_grad()
        lower_bound, discrim_loss, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2 = model(
            features, idxs, len(train_loader.dataset), nsegs
        )
        tr_summ_vars = (log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2)
        loss = loss_function(lower_bound, discrim_loss, args.alpha_dis)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if args.legacy and batch_idx + 1 % args.log_interval == 0:
            current_pos = batch_idx * len(features)
            tot_len = len(train_loader.dataset)
            percentage = 100.0 * batch_idx / len(train_loader)
            cur_loss = loss.item() / len(features)

            print(
                f"====> Train Epoch: {epoch} [{current_pos}/{tot_len} ({percentage:.0f}%)]\tLoss: {cur_loss:.6f}"
            )
        if np.isnan(lower_bound):
            print("Training diverged")
            raise sys.exit(2)
        if args.legacy and batch_idx + 1 % args.steps_per_epoch == 0:
            break

    train_loss /= len(train_loader.dataset)
    print(f"====> Train set average loss: {train_loss:.4f}")

    # eval
    model.eval()
    val_loss = 0.0
    if summary_list is None:
        summary_list = [0, 0, 0, 0, 0]
    with torch.no_grad():
        for (key, feature, nsegs) in val_loader:
            feature = feature.to(device)
            val_lower_bound, _, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2 = model(
                feature, key, len(train_loader.dataset), nsegs
            )
            val_summ_vars = log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2
            val_loss += loss_function(lower_bound, discrim_loss, args.alpha_dis).item()
            summary_list = [
                map(torch.sum, summary_list, (val_lower_bound, *val_summ_vars))
            ]
            current_pos = batch_idx * len(feature)
            tot_len = len(val_loader.dataset)
            percentage = 100.0 * batch_idx / len(val_loader)
            cur_loss = loss.item() / len(feature)

            print(
                f"====> Validation Epoch: {epoch} [{current_pos}/{tot_len} ({percentage:.0f}%)]\tLoss: {cur_loss:.6f}"
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

    if check_best(val_lower_bound, best_val_lb):
        best_epoch = epoch

    save_checkpoint(
        model,
        optimizer,
        summary_list,
        values,
        base_string,
        epoch,
        best_epoch,
        val_lower_bound,
        best_val_lb,
        exp_dir,
    )

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}"
    )

    if check_terminate(epoch, best_epoch, args.patience, args.epochs):
        print("Training terminated!")
        break
    summary_list = None

print("Training complete!")
