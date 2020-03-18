import os
import argparse
from sphfile import SPHFile
from .utils import maybe_makedir


def process_timit(
    root_dir: str,
    output_dir: str = "./datasets/timit_np_fbank",
    dev_spk_path: str = "./misc/timit_dev_spk.list",
    test_spk_path: str = "./misc/timit_test_spk.list",
) -> None:
    """Generates .scp files for the TIMIT dataset

    Args:
        root_dir:      Directory holding the dataset
        output_dir:    Where .scp files should be saved
        dev_spk_path:  Path to a file containing all the speakers for the dev set
        test_spk_path: Path to a file containing all the speakers for the train set
    """
    # retrieve partition
    with open(dev_spk_path) as f:
        dt_spks = [line.rstrip().lower() for line in f]
    with open(test_spk_path) as f:
        tt_spks = [line.rstrip().lower() for line in f]

    # convert sph to wav and dump scp
    wav_dir = os.path.abspath(f"{output_dir}/wav")
    tr_scp = f"{output_dir}/train/wav.scp"
    dt_scp = f"{output_dir}/dev/wav.scp"
    tt_scp = f"{output_dir}/test/wav.scp"

    for file in (wav_dir, tr_scp, dt_scp, tt_scp):
        maybe_makedir(file)

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
                sph_path = f"{root}/{fname}"
                path = f"{wav_dir}/{spk}_{fname}"
                uttid = f"{spk}_{os.path.splitext(fname)[0]}"
                f.write(f"{uttid} {path}\n")
                sph = SPHFile(sph_path)
                sph.write_wav(path)

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
        help="feature type",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./datasets/timit_np_fbank",
        help="output data directory",
    )
    parser.add_argument(
        "--dev_spk",
        type=str,
        default="./misc/timit_dev_spk.list",
        help="path to list of dev set speakers",
    )
    parser.add_argument(
        "--test_spk",
        type=str,
        default="./misc/timit_test_spk.list",
        help="path to list of test set speakers",
    )
    args = parser.parse_args()
    print(args)

    process_timit(args.timit_dir, args.out_dir, args.dev_spk, args.test_spk)
