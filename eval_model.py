import os
import sys
import time
import argparse
import torch
from utils import load_args, create_training_strings
from loggers import VisdomLogger, TensorBoardLogger

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("exp_dir", type=str, help="Experiment directory")
parser.add_argument(
    "--set-name",
    type=str,
    default="dev",
    choices=["train", "dev", "test"],
    help="Name of dataset partition to evaluate",
)
parser.add_argument(
    "--seqlist", type=str, default=None, help="Specify a list of sequences to evaluate"
)
parser.add_argument(
    "--step", type=int, default=-1, help="Step of the model to load. -1 loads best step"
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
args = parser.parse_args()

loaded_args = load_args(args.exp_dir)

_, _, run_id = create_training_strings(args)

if args.visdom:
    visdom_logger = VisdomLogger(run_id, loaded_args.epochs)
if args.tensorboard:
    tensorboard_logger = TensorBoardLogger(run_id, args.tb_log_dir, args.log_params)
