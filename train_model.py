import argparse
import logging
import os
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default="timit", help="dataset to use")
parser.add_argument(
    "--is_numpy",
    action="store_true",
    dest="is_numpy",
    help="dataset format; kaldi by default",
)
parser.add_argument(
    "--model",
    type=str,
    default="fhvae",
    help="model architecture; {fhvae|simple_fhvae}",
)
parser.add_argument(
    "--alpha_dis", type=float, default=10.0, help="discriminative objective weight"
)
parser.add_argument(
    "--n_epochs", type=int, default=100, help="number of maximum training epochs"
)
parser.add_argument(
    "--n_patience",
    type=int,
    default=10,
    help="number of maximum consecutive non-improving epochs",
)
parser.add_argument(
    "--n_steps_per_epoch",
    type=int,
    default=5000,
    help="number of training steps per epoch",
)
parser.add_argument(
    "--n_print_steps", type=int, default=200, help="number of steps to print statistics"
)
args = parser.parse_args()
print(args)

# load data

# if args.dataset == 'timit':
#     preprocess_timit()
# else:
#     preprocess_librispeech()
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
