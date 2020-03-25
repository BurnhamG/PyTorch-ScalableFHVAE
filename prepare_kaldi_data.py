import os
import argparse
import subprocess
from typing import Tuple
from pathlib import Path


def prepare_kaldi(
    wav_scp: str,
    feat_ark: str = None,
    feat_scp: str = None,
    len_scp: str = None,
    fbank_conf="./misc/fbank.conf",
    kaldi_root="./kaldi",
) -> Tuple[Path, Path, Path]:
    out_files = [feat_ark, feat_scp, len_scp]
    filenames = ("feats.ark", "feats.scp", "len.scp")

    # If these names are not provided, save them to the same location as the wav.scp
    for p, name in zip(out_files, filenames):
        if p is None:
            p = os.path.join(os.path.dirname(wav_scp), name)
        os.makedirs(os.path.dirname(p), exist_ok=True)

    feat_comp_cmd = [
        os.path.join(kaldi_root, "src/bin/compute-fbank-feats"),
        f"scp,p:{wav_scp}",
        f"ark,scp:{feat_ark},{feat_scp}",
        f"--config={fbank_conf}",
    ]
    feat_compute = subprocess.Popen(feat_comp_cmd)

    # The next operation requires completion of this first command
    feat_compute.wait()

    feat_len_cmd = [
        os.path.join(kaldi_root, "src/featbin/feat-to-len"),
        f"scp:{feat_scp}",
        f"ark,t:{len_scp}",
    ]
    feat_to_len = subprocess.Popen(feat_len_cmd)

    feat_to_len.wait()

    exit_codes = [p.returncode for p in (feat_compute, feat_to_len)]
    if any(exit_codes):
        commands = (" ".join(feat_comp_cmd), " ".join(feat_len_cmd))
        error_info = [
            f"{cmd}: {error}" for (cmd, error) in zip(commands, exit_codes) if error > 0
        ]
        raise RuntimeError(f"Non-zero return code(s): {', '.join(error_info)}")
    return Path(feat_ark), Path(feat_scp), Path(len_scp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("wav_scp", type=str, help="Input wav scp file")
    parser.add_argument(
        "feat_ark", type=str, default=None, help="Output feats.ark file"
    )
    parser.add_argument(
        "feat_scp", type=str, default=None, help="Output feats.scp file"
    )
    parser.add_argument("len_scp", type=str, default=None, help="Output len.scp file")
    parser.add_argument(
        "--fbank_conf",
        type=str,
        default="./misc/fbank.conf",
        help="Kaldi fbank configuration",
    )
    parser.add_argument(
        "--kaldi_root", type=str, default="./kaldi", help="Kaldi root directory",
    )
    args = parser.parse_args()
    print(args)

    prepare_kaldi(
        args.wav_scp,
        args.feat_ark,
        args.feat_scp,
        args.len_scp,
        args.fbank_conf,
        args.kaldi_root,
    )
