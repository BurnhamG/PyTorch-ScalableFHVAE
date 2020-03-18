from utils import maybe_makedir, AudioUtils
import argparse
import librosa
import numpy as np
import os
import sys
import time


def prepare_numpy(
    dataset: str,
    save_feats: bool = True,
    np_dir: str = None,
    ftype: str = "fbank",
    sample_rate: int = None,
    win_t: float = 0.025,
    hop_t: float = 0.010,
    n_mels: int = 80,
):
    root_dir = os.path.abspath(f"./datasets/{dataset}")
    if save_feats:
        if np_dir is None:
            np_dir = os.path.abspath(f"./datasets/{dataset.strip()}/numpy/{ftype}")
        maybe_makedir(np_dir)
        for set in ['train','dev','test']:
            set_path = os.path.join(np_path,set)
            maybe_makedir(os.path.dirname


    librosa.load(path, sample_rate, mono=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("wav_scp", type=str, help="input wav scp file")
    parser.add_argument("np_dir", type=str, help="output directory for numpy matrices")
    parser.add_argument("feat_scp", type=str, help="output feats.scp file")
    parser.add_argument("len_scp", type=str, help="output len.scp file")
    parser.add_argument(
        "--ftype",
        type=str,
        default="fbank",
        choices=["fbank", "spec"],
        help="feature type to compute",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="resample raw audio to specified value if not None",
    )
    parser.add_argument(
        "--win_t", type=float, default=0.025, help="window size in second"
    )
    parser.add_argument(
        "--hop_t", type=float, default=0.010, help="frame spacing in second"
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=80,
        help="number of filter banks if choosing fbank",
    )
    args = parser.parse_args()
    print(args)

    reader = lambda path: librosa.load(path, args.sr, mono=True)
    if args.ftype == "fbank":
        mapper = lambda y, sr: np.transpose(
            AudioUtils.to_melspec(
                y, sr, int(sr * args.win_t), args.hop_t, args.win_t, n_mels=args.n_mels
            )
        )
    elif args.ftype == "spec":
        mapper = lambda y, sr: np.transpose(
            AudioUtils.rstft(y, sr, int(sr * args.win_t), args.hop_t, args.win_t)
        )

    prepare_numpy(
        args.wav_scp, args.np_dir, args.feat_scp, args.len_scp, reader, mapper
    )
