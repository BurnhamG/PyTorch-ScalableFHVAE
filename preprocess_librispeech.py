import os
import argparse
from .utils import maybe_makedir


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("librispeech_dir", type=str, help="LibriSpeech raw data directory")
parser.add_argument(
    "--ftype", type=str, default="fbank", choices=["fbank", "spec"], help="feature type"
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="./datasets/librispeech_np_fbank",
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


# dump wav scp
def find_audios(d):
    uid_path = []
    for root, _, fnames in sorted(os.walk(d)):
        for fname in fnames:
            if fname.endswith(".flac") or fname.endswith(".FLAC"):
                uid_path.append((os.path.splitext(fname)[0], f"{root}/{fname}"))
    return sorted(uid_path, key=lambda x: x[0])


def write_scp(out_path, set_list):
    maybe_makedir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        uid_path = []
        for set in set_list:
            uid_path += find_audios(f"{args.librispeech_dir}/{set}")
        for uid, path in uid_path:
            f.write(f"{uid} {path}\n")


write_scp(f"{args.out_dir}/train/wav.scp", args.train_list)
write_scp(f"{args.out_dir}/dev/wav.scp", args.dev_list)
write_scp(f"{args.out_dir}/test/wav.scp", args.test_list)

print("generated wav scp")
