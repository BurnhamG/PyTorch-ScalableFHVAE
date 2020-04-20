import os
import time
import itertools
import argparse
from typing import List, Tuple
from pathlib import Path
from pydub import AudioSegment
from multiprocessing import Pool
import math


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


def convert_audios(filelist):
    converted_files = []
    list_len = len(filelist)
    stime = time.time()
    for idx, (uid, file) in enumerate(filelist):
        audio = AudioSegment.from_file(file, "flac")
        new_filepath = os.path.splitext(file)[0] + ".wav"
        audio.export(new_filepath, "wav")
        converted_files.append((uid, new_filepath))
        if (idx + 1) % 100 == 0:
            print(
                f"PID {os.getpid()} converted {idx + 1} files out of {list_len} in {time.time() - stime:.2f} seconds"
            )
    return converted_files


def write_scp(
    root_dir: Path, out_path: Path, subset_list: list, data_format: str = "numpy"
) -> None:
    """Writes uid and audio path to Kaldi .scp file"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        uid_path = []
        for se in subset_list:
            if os.path.exists(root_dir / f"{se}"):
                uid_path += find_audios(root_dir / f"{se}")
                if data_format == "kaldi":
                    n = math.ceil(len(uid_path) / 8)
                    uid_path_lists = (
                        uid_path[i : i + n] for i in range(0, len(uid_path), n)
                    )
                    print(f"Converting {len(uid_path)} utterances to .wav for Kaldi")
                    with Pool(8) as p:
                        results = p.imap(convert_audios, uid_path_lists)
                        uid_path = sorted(
                            list(itertools.chain.from_iterable(results)),
                            key=lambda x: x[0],
                        )
                    for uid, path in uid_path:
                        f.write(f"{uid} {path.replace('.flac','.wav')}\n")
                else:
                    for uid, path in uid_path:
                        f.write(f"{uid} {path}\n")


def process_librispeech(
    raw_data_dir: Path,
    output_dir: Path,
    data_format: str = "numpy",
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
    set_names = ("train", "dev", "test")

    # avoid mutable default args
    if train_list is None:
        train_list = ["train-clean-100"]
    if dev_list is None:
        dev_list = ["dev-clean", "dev-other"]
    if test_list is None:
        test_list = ["test-clean", "dev-other"]

    train_scp, dev_scp, test_scp = [output_dir / f"{se}/wav.scp" for se in set_names]

    for scp, subset_list, set_name in zip(
        [train_scp, dev_scp, test_scp], [train_list, dev_list, test_list], set_names
    ):
        print(f"{set_name.capitalize()}:")
        write_scp(raw_data_dir, scp, subset_list, data_format)

    print("Generated scp files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("raw_data_dir", type=str, help="LibriSpeech raw data directory")
    parser.add_argument("output_dir", type=str, help="Directory for data output")
    parser.add_argument(
        "--data-format",
        type=str,
        default="numpy",
        choices=["numpy", "kaldi"],
        help="Data format to use",
    )
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
        default=["test-clean", "test-other"],
        help="Test sets to include {test-clean, test-other}",
    )
    args = parser.parse_args()
    print(args)

    process_librispeech(
        Path(args.raw_data_dir),
        Path(args.output_dir),
        args.data_format,
        args.train_list,
        args.dev_list,
        args.test_list,
    )
