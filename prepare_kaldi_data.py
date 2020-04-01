import os
import time
import argparse
import subprocess
from typing import Tuple
from pathlib import Path


def prepare_kaldi(
    wav_scp: str,
    fbank_conf: str = "./misc/fbank.conf",
    kaldi_root: str = "./kaldi",
    set_name: str = "",
) -> Tuple[int, Tuple[Path, Path, Path, Path]]:
    """Handles Kaldi format feature and script file generation and saving


    Args:
        wav_scp:    Input wav.scp file
        fbank_conf: Location of the fbank.conf file for Kaldi to use in feature computation
        kaldi_root: Kaldi root directory

    """
    if set_name != "":
        set_name += ": "

    filenames = ("feats.ark", "feats.scp", "len.scp")

    file_paths = [os.path.join(os.path.dirname(wav_scp), name) for name in filenames]

    feat_ark, feat_scp, len_scp = file_paths

    feat_comp_cmd = [
        os.path.join(kaldi_root, "src/featbin/compute-fbank-feats"),
        f"--config={fbank_conf}",
        f"scp,p:{wav_scp}",
        f"ark,scp:{feat_ark},{feat_scp}",
    ]
    feat_compute = subprocess.Popen(
        feat_comp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stime = time.time()
    count = 0
    for i, line in enumerate(feat_compute.stderr):
        msg = line.decode()
        # Kaldi only logs every 10 files, so this logs every 200 files
        if i % 20 == 0 and i > 0:
            processed_idx = msg.find("Processed")
            print(
                f"{set_name.capitalize():7}{' '.join(msg[processed_idx:].split()[1:])} in {time.time() - stime:.2f} seconds"
            )
        count = i + 1

    # The next operation requires completion of this first command
    feat_compute.wait()

    print(
        f"{set_name.capitalize()} feature computation completed in {time.time()-stime:.2f} seconds"
    )

    feat_len_cmd = [
        os.path.join(kaldi_root, "src/featbin/feat-to-len"),
        f"scp:{feat_scp}",
        f"ark,t:{len_scp}",
    ]
    feat_to_len = subprocess.Popen(
        feat_len_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    feat_to_len.wait()

    exit_codes = [p.returncode for p in (feat_compute, feat_to_len)]
    if any(exit_codes):
        commands = (" ".join(feat_comp_cmd), " ".join(feat_len_cmd))
        error_info = [
            f"{cmd}: {error}" for (cmd, error) in zip(commands, exit_codes) if error > 0
        ]
        raise RuntimeError(f"Non-zero return code(s): {', '.join(error_info)}")
    return count, (Path(wav_scp), Path(feat_ark), Path(feat_scp), Path(len_scp))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("wav_scp", type=str, help="Input wav scp file")
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
        args.wav_scp, args.fbank_conf, args.kaldi_root,
    )
