import os
import argparse
from typing import List, Tuple


# dump wav scp
def find_audios(dir: str) -> List[Tuple[str, str]]:
    """Find .flac files in the given directory

    Args:
        dir: Directory to search

    Returns:
        Sorted list of (audio_identifier, path_to_file) for all files found

    """
    uid_path = []
    for file in sorted(os.listdir(dir)):
        if file.lower().endswith(".flac"):
            uid_path.append((os.path.splitext(file)[0], os.path.join(dir, file)))
    return sorted(uid_path, key=lambda x: x[0])


def write_scp(root_dir: str, out_path: str, set_list: list) -> None:
    """Writes uid and audio path to Kaldi .scp file"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        uid_path = []
        for set in set_list:
            uid_path += find_audios(f"{root_dir}/{set}")
        for uid, path in uid_path:
            f.write(f"{uid} {path}\n")


def process_librispeech(
    raw_data_dir: str,
    output_dir: str = "./datasets/librispeech",
    train_list: list = None,
    dev_list: list = None,
    test_list: list = None,
) -> None:
    """Generates .scp files for the Librispeech dataset

    Args:
        raw_data_dir: Base directory
        output_dir:   Where .scp files should be saved
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

    write_scp(raw_data_dir, f"{output_dir}/train/wav.scp", train_list)
    write_scp(raw_data_dir, f"{output_dir}/dev/wav.scp", dev_list)
    write_scp(raw_data_dir, f"{output_dir}/test/wav.scp", test_list)

    print("generated wav scp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "librispeech_dir", type=str, help="LibriSpeech raw data directory"
    )
    parser.add_argument(
        "--ftype",
        type=str,
        default="fbank",
        choices=["fbank", "spec"],
        help="Feature type",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./datasets/librispeech",
        help="Base output data directory",
    )
    parser.add_argument(
        "--train_list",
        type=str,
        nargs="*",
        default=["train-clean-100"],
        help="Train sets to include {train-clean-100, train-clean-360, train-other-500}",
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
        args.librispeech_dir,
        args.out_dir,
        args.train_list,
        args.dev_list,
        args.test_list,
    )
