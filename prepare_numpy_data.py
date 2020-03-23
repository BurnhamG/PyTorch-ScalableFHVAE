from utils import AudioUtils
import argparse
import librosa
import numpy as np
import os
import time
import contextlib
from multiprocessing import Pool
from nptyping import Array
from typing import Dict


def generate_feat(
    ftype: str,
    audio_data: Array[float],
    sample_rate: int,
    win_t: float,
    hop_t: float,
    n_mels: int,
) -> Array[float]:
    """Generates the features for an audio sample

    Args:
        ftype:       Type of computed feature
        audio_data:  Input audio sample
        sample_rate: Audio sample rate
        win_t:       FFT window size in seconds
        hop_t:       Frame spacing in seconds
        n_mels:      Number of filter banks if using 'fbank' as the computed feature

    """
    if ftype == "fbank":
        feat = np.transpose(
            AudioUtils.to_melspec(
                audio_data,
                sample_rate,
                int(sample_rate * win_t),
                hop_t,
                win_t,
                n_mels=n_mels,
            )
        )
    else:
        feat = np.transpose(
            AudioUtils.rstft(
                audio_data, sample_rate, int(sample_rate * win_t), hop_t, win_t
            )
        )
    return feat


def save_feats(
    dataset: str,
    set_name: str,
    ftype: str,
    sample_rate: int,
    win_t: float,
    hop_t: float,
    n_mels: int,
    paths: Dict[str, str] = None,
) -> int:
    """Saves the generated features and script files for a dataset

    Args:
        dataset:     Name of the dataset (i.e. librispeech) being used
        set_name:    Name of the set (train, dev, test) to operate on
        ftype:       Type of computed feature
        sample_rate: Sample rate for resampling if not None
        win_t:       FFT window size in seconds
        hop_t:       Frame spacing in seconds
        n_mels:      Number of filter banks if using 'fbank' as the computed feature
        paths:       Dictionary of (optional) paths for saving files

    Returns:
        Number of files that were processed

    """
    if paths is None:
        paths = {}

    root_dir = os.path.abspath(f"./datasets/{dataset}")
    if paths.get("output_dir") is not None:
        set_path = paths.get("output_dir")
    else:
        set_path = os.path.join(root_dir, set_name)
    if paths.get("wav_scp") is not None:
        wav_path = paths.get("wav_scp")
    else:
        wav_path = os.path.join(set_path, "wav.scp")
    if paths.get("feat_scp") is not None:
        feat_path = paths.get("feat_scp")
    else:
        feat_path = os.path.join(set_path, "feats.scp")
    if paths.get("len_scp") is not None:
        len_path = paths.get("len_scp")
    else:
        len_path = os.path.join(set_path, "len.scp")

    set_start_time = time.time()
    count = 0
    with contextlib.ExitStack() as stack:
        wavfile = stack.enter_context(open(wav_path))
        featfile, lenfile = [
            stack.enter_context(open(f, "w")) for f in [feat_path, len_path]
        ]
        for i, line in enumerate(wavfile):
            seq, path = line.rstrip().split()
            y, _sr = librosa.load(path, sample_rate, mono=True)
            if sample_rate is None:
                sample_rate = _sr
            elif sample_rate != _sr:
                raise ValueError(f"Inconsistent sample rate ({sample_rate} != {_sr}.")

            feat = generate_feat(ftype, y, sample_rate, win_t, hop_t, n_mels)
            np_path = os.path.join(set_path, "numpy", f"{seq}.npy")
            with open(np_path, "wb") as numpyfile:
                np.save(numpyfile, feat)
            featfile.write(f"{seq} {np_path}\n")
            lenfile.write(f"{seq} {len(feat)}\n")
            count = i + 1
            if (count) % 1000 == 0:
                print(
                    f"{count} files in {set_name} set over {time.time() - set_start_time} seconds."
                )
    return count


def prepare_numpy(
    dataset: str,
    ftype: str = "fbank",
    sample_rate: int = None,
    win_t: float = 0.025,
    hop_t: float = 0.010,
    n_mels: int = 80,
    **kwargs,
) -> None:
    """Handles feature and script file generation and saving

    Args:
        dataset:     Name of the dataset for which features are generated
        ftype:       Type of computed feature
        sample_rate: Sample rate for resampling if not None
        win_t:       FFT window size in seconds
        hop_t:       Frame spacing in seconds
        n_mels:      Number of filter banks if using 'fbank' as the computed feature

    """
    paths = {}
    for pathvar in ["wave_scp", "output_dir", "feat_scp", "len_scp"]:
        if pathvar in kwargs:
            paths[pathvar] = kwargs[pathvar]

    for path in paths:
        os.makedirs(paths[path], exist_ok=True)

    starmap_args = []
    for s in ["train", "dev", "test"]:
        func_args = [dataset, s, ftype, sample_rate, win_t, hop_t, n_mels, paths]
        starmap_args.append(tuple(func_args))

    files_start_time = time.time()
    with Pool(3) as p:
        results = p.starmap(save_feats, starmap_args)

    print(
        f"Processed {sum(results)} files in {time.time() - files_start_time} seconds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("wav_scp", type=str, help="input wav scp file")
    parser.add_argument("np_dir", type=str, help="output directory for numpy matrices")
    parser.add_argument("feat_scp", type=str, help="output feats.scp file")
    parser.add_argument("len_scp", type=str, help="output len.scp file")
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispech",
        choices=["librispeech", "timit"],
        help="Dataset name",
    )
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

    prepare_numpy(
        args.dataset,
        args.ftype,
        args.sr,
        args.win_t,
        args.hop_t,
        args.n_mels,
        wav_scp=args.wav_scp,
        feat_scp=args.feat_scp,
        len_scp=args.len_scp,
        output_dir=args.np_dir,
    )
