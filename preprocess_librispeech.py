import os
import argparse
from typing import List, Tuple
from pathlib import Path
from utils import create_output_dir


# dump wav scp
def find_audios(dir: Path) -> List[Tuple[str, str]]:
    """Find .flac files in the given directory

    Args:
        dir: Directory to search

    Returns:
        Sorted list of (audio_identifier, path_to_file) for all files found

    """
    uid_path = []
    for root, _, files in sorted(os.walk(dir)):
        for file in files:
            if file.lower().endswith(".flac"):
                uid_path.append((os.path.splitext(file)[0], os.path.join(root, file)))
    return sorted(uid_path, key=lambda x: x[0])


def write_scp(root_dir: Path, out_path: Path, set_list: list) -> None:
    """Writes uid and audio path to Kaldi .scp file"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        uid_path = []
        for se in set_list:
            if os.path.exists(root_dir / f"{se}"):
                uid_path += find_audios(root_dir / f"{se}")
        print(uid_path, "UID_PATH")
        for uid, path in uid_path:
            f.write(f"{uid} {path}\n")


def process_librispeech(
    raw_data_dir: Path,
    output_dir: Path,
    train_list: list = None,
    dev_list: list = None,
    test_list: list = None,
) -> Tuple[Path, Path, Path]:
    """Generates .scp files for the Librispeech dataset

    Args:
        raw_data_dir: Base directory
        train_list:   Training sets to process
        dev_list:     Development sets to process
        test_list:    Test sets to process

    """
    # avoid mutable default args
    if train_list is None:
        train_list = ["train-clean-100"]
    if dev_list is None:
        dev_list = ["dev-clean", "dev-other"]
    if test_list is None:
        test_list = ["test-clean", "dev-other"]

    train_scp, dev_scp, test_scp = [
        output_dir / f"{se}/wav.scp" for se in ["train", "dev", "test"]
    ]

    for scp, set_list in zip(
        [train_scp, dev_scp, test_scp], [train_list, dev_list, test_list]
    ):
        write_scp(raw_data_dir, scp, set_list)

    print("Generated scp files")

    return train_scp, dev_scp, test_scp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("raw_data_dir", type=str, help="LibriSpeech raw data directory")
    parser.add_argument("output_dir", type=str, help="Directory for data output")
    parser.add_argument(
        "--train_list",
        type=str,
        nargs="*",
        default=["train-clean-100"],
        help="Training sets to include {train-clean-100, train-clean-360, train-other-500}",
    )
    parser.add_argument(
        "--dev_list",
        type=str,
        nargs="*",
        default=["dev-clean", "dev-other"],
        help="Dev sets to include {dev-clean, dev-other}",
    )
    parser.add_argument(
        "--test_list",
        type=str,
        nargs="*",
        default=["test-clean", "dev-other"],
        help="Test sets to include {test-clean, test-other}",
    )
    args = parser.parse_args()
    print(args)

    process_librispeech(
        args.raw_data_dir,
        args.output_dir,
        args.train_list,
        args.dev_list,
        args.test_list,
    )
