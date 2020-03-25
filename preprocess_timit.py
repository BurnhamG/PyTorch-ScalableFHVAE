import os
import argparse
from sphfile import SPHFile
from pathlib import Path
from utils import create_output_dir


def process_timit(
    root_dir: str,
    feat_type: str = "fbank",
    data_format: str = "numpy",
    dev_spk_path: str = "./misc/timit_dev_spk.list",
    test_spk_path: str = "./misc/timit_test_spk.list",
) -> None:
    """Generates .scp files for the TIMIT dataset

    Args:
        root_dir:      Directory holding the dataset
        feat_type:     Type of feature that will be computed
        data_format:   Format used for storing computed features
        dev_spk_path:  Path to a file containing all the speakers for the dev set
        test_spk_path: Path to a file containing all the speakers for the train set

    """
    # retrieve partition
    with open(Path(dev_spk_path)) as f:
        dt_spks = [line.rstrip().lower() for line in f]
    with open(Path(test_spk_path)) as f:
        tt_spks = [line.rstrip().lower() for line in f]

    output_dir = create_output_dir(root_dir, data_format, feat_type)

    # convert sph to wav and dump scp
    wav_dir = Path(os.path.abspath(output_dir / "wav"))
    tr_scp = output_dir / "train/wav.scp"
    dt_scp = output_dir / "dev/wav.scp"
    tt_scp = output_dir / "test/wav.scp"

    for file in (wav_dir, tr_scp, dt_scp, tt_scp):
        os.makedirs(file, exist_ok=True)

    tr_f = open(tr_scp, "w")
    dt_f = open(dt_scp, "w")
    tt_f = open(tt_scp, "w")

    for root, _, fnames in sorted(os.walk(root_dir)):
        spk = root.split("/")[-1].lower()
        if spk in dt_spks:
            f = dt_f
        elif spk in tt_spks:
            f = tt_f
        else:
            f = tr_f

        for fname in fnames:
            if fname.endswith(".wav") or fname.endswith(".WAV"):
                # WAV files are SPHERE-headered
                sph_path = Path(f"{root}/{fname}")
                path = wav_dir / f"{spk}_{fname}"
                uttid = f"{spk}_{os.path.splitext(fname)[0]}"
                f.write(f"{uttid} {str(path)}\n")
                sph = SPHFile(str(sph_path))
                sph.write_wav(str(path))

    tr_f.close()
    dt_f.close()
    tt_f.close()

    print("Converted to wav and dumped .scp files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("timit_dir", type=str, help="TIMIT raw data directory")
    parser.add_argument(
        "--ftype",
        type=str,
        default="fbank",
        choices=["fbank", "spec"],
        help="Feature type",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="numpy",
        choices=["kaldi", "numpy"],
        help="Format used to store data.",
    )
    parser.add_argument(
        "--dev_spk",
        type=str,
        default="./misc/timit_dev_spk.list",
        help="Path to list of dev set speakers",
    )
    parser.add_argument(
        "--test_spk",
        type=str,
        default="./misc/timit_test_spk.list",
        help="Path to list of test set speakers",
    )
    args = parser.parse_args()
    print(args)

    process_timit(
        args.timit_dir, args.ftype, args.data_format, args.dev_spk, args.test_spk
    )
