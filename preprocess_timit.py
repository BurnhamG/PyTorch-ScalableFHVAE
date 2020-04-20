import os
import argparse
from sphfile import SPHFile
from pathlib import Path
from utils import create_output_dir
from typing import Tuple


def process_timit(
    raw_data_dir: Path,
    output_dir: Path,
    dev_spk_path: str = "./misc/timit_dev_spk.list",
    test_spk_path: str = "./misc/timit_test_spk.list",
) -> Tuple[Path, Path, Path]:
    """Generates .scp files for the TIMIT dataset

    Args:
        raw_data_dir:      Directory holding the dataset
        output_dir:
        dev_spk_path:  Path to a file containing all the speakers for the dev set
        test_spk_path: Path to a file containing all the speakers for the train set

    """
    # retrieve partition
    with open(Path(dev_spk_path)) as f:
        dt_spks = [line.rstrip().lower() for line in f]
    with open(Path(test_spk_path)) as f:
        tt_spks = [line.rstrip().lower() for line in f]

    # convert sph to wav and dump scp
    wav_dir = output_dir / "wav"
    train_scp, dev_scp, test_scp = [
        output_dir / f"{se}/wav.scp" for se in ["train", "dev", "test"]
    ]

    for file in (wav_dir, train_scp, dev_scp, test_scp):
        os.makedirs(file, exist_ok=True)

    tr_f = open(train_scp, "w")
    dt_f = open(dev_scp, "w")
    tt_f = open(test_scp, "w")

    for root, _, fnames in sorted(os.walk(raw_data_dir)):
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
    parser.add_argument("raw_data_dir", type=str, help="TIMIT raw data directory")
    parser.add_argument("output_dir", type=str, help="Directory for data output")
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

    process_timit(args.raw_data_dir, args.output_dir, args.dev_spk, args.test_spk)
