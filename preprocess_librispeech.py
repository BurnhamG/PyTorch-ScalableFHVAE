import os
import argparse
from .utils import maybe_makedir


# dump wav scp
def find_audios(dir: str) -> list:
    """Find .flac files in the given directory"""
    uid_path = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith(".flac") or fname.endswith(".FLAC"):
                uid_path.append((os.path.splitext(fname)[0], f"{root}/{fname}"))
    return sorted(uid_path, key=lambda x: x[0])


def write_scp(root_dir: str, out_path: str, set_list: list) -> None:
    maybe_makedir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        uid_path = []
        for set in set_list:
            uid_path += find_audios(f"{root_dir}/{set}")
        for uid, path in uid_path:
            f.write(f"{uid} {path}\n")


def process_librispeech(
    root_dir: str,
    output_dir: str = "./datasets/librispeech",
    train_list: list = None,
    dev_list: list = None,
    test_list: list = None,
) -> None:
    """Generates .scp files for the Librispeech dataset

    Args:
        root_dir:   Base directory
        output_dir: Where .scp files should be saved
        train_list: Training sets to process
        dev_list:   Development sets to process
        test_list:  Test sets to process
    """
    # avoid mutable default args
    if train_list is None:
        train_list = ["train-clean-100"]
    if dev_list is None:
        dev_list = ["dev-clean", "dev-other"]
    if test_list is None:
        test_list = ["test-clean", "dev-other"]

    write_scp(root_dir, f"{output_dir}/train/wav.scp", train_list)
    write_scp(root_dir, f"{output_dir}/dev/wav.scp", dev_list)
    write_scp(root_dir, f"{output_dir}/test/wav.scp", test_list)

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
        help="feature type",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./datasets/librispeech",
        help="output data directory",
    )
    parser.add_argument(
        "--train_list",
        type=str,
        nargs="*",
        default=["train-clean-100"],
        help="train sets to include {train-clean-100, train-clean-360, train-other-500}",
    )
    parser.add_argument(
        "--dev_list",
        type=str,
        nargs="*",
        default=["dev-clean", "dev-other"],
        help="dev sets to include {dev-clean, dev-other}",
    )
    parser.add_argument(
        "--test_list",
        type=str,
        nargs="*",
        default=["test-clean", "dev-other"],
        help="test sets to include {test-clean, test-other}",
    )
    args = parser.parse_args()
    print(args)

    process_librispeech(args.out_dir, args.train_list, args.dev_list, args.test_list)
