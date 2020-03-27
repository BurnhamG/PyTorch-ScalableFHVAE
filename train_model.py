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
    default="./checkpoints",
    type=str,
    help="Directory that will hold the model checkpoints",
)
args = parser.parse_args()
print(args)

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

if args.visdom:
    visdom_logger = VisdomLogger(run_id, args.n_epochs)
if args.tensorboard:
    tensorboard_logger = TensorBoardLogger(run_id, args.log_dir, args.log_params)

os.makedirs(args.checkpoint_dir, exist_ok=True)

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

model.to(device)
# set up experiment directory
exp_root = Path("./experiments") / base_string
exp_dir = exp_root / exp_string
os.makedirs(exp_dir, exist_ok=True)

# run training
optimizer = Adam(
    model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2)
)

for epoch in range(args.n_epochs):
    # training
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(audio_loader):
        data = data.to(args.device)
        optimizer.zero_grad()
        lower_bound, discrim_loss = model(*data, len(audio_dataset))
        loss = loss_function(lower_bound, discrim_loss, args.alpha_dis)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx + 1 % args.log_interval == 0:
            current_pos = batch_idx * len(data)
            tot_len = len(audio_loader.dataset)
            percentage = 100.0 * batch_idx / len(audio_loader)
            cur_loss = loss.item() / len(data)

            print(
                f"Train Epoch: {epoch} [{current_pos}/{tot_len} ({percentage:.0f}%)]\tLoss: {cur_loss:.6f}"
            )

    values = {
        "loss_results": loss_results,
        "lower_bound_results": lower_bound_results,
        "discrim_loss_results": discrim_loss_results,
    }
    if args.tensorboard:
        tensorboard_logger.update(epoch, values, model.named_parameters())
    if args.visdom:
        visdom_logger.update(epoch, values)

    # Save model
    checkpoint = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
    }
    save_ckp(checkpoint, is_best, args.checkpoint_dir, model_dir)

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(audio_loader.dataset):.4f}"
    )

    # eval
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(dev_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(lower_bound, discrim_loss, args.alpha_dis).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]]
                )
                save_image(
                    comparison.cpu(),
                    "results/reconstruction_" + str(epoch) + ".png",
                    nrow=n,
                )

    test_loss /= len(dev_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))


# alpha/discriminative weight of 10 was found to produce best results
def loss_function(lower_bound, log_qy, alpha=10.0):
    """Discriminative segment variational lower bound

    Segment variational lower bound plus the (weighted) discriminative objective

    """

    return -1 * torch.mean(lower_bound + alpha * log_qy)


import shutil


def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir / "checkpoint.pt"
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir / "best_model.pt"
        shutil.copyfile(f_path, best_fpath)
