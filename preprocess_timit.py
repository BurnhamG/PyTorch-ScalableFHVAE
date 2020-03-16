import os
import argparse
from sphfile import SPHFile
from .utils import maybe_makedir


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("timit_dir", type=str, help="TIMIT raw data directory")
parser.add_argument(
    "--ftype", type=str, default="fbank", choices=["fbank", "spec"], help="feature type"
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

# retrieve partition
with open(args.dev_spk) as f:
    dt_spks = [line.rstrip().lower() for line in f]
with open(args.test_spk) as f:
    tt_spks = [line.rstrip().lower() for line in f]

# convert sph to wav and dump scp
wav_dir = os.path.abspath(f"{args.out_dir}/wav")
tr_scp = f"{args.out_dir}/train/wav.scp"
dt_scp = f"{args.out_dir}/dev/wav.scp"
tt_scp = f"{args.out_dir}/test/wav.scp"

maybe_makedir(wav_dir)
maybe_makedir(os.path.dirname(tr_scp))
maybe_makedir(os.path.dirname(dt_scp))
maybe_makedir(os.path.dirname(tt_scp))

tr_f = open(tr_scp, "w")
dt_f = open(dt_scp, "w")
tt_f = open(tt_scp, "w")

paths = []
for root, _, fnames in sorted(os.walk(args.timit_dir)):
    spk = root.split("/")[-1].lower()
    if spk in dt_spks:
        f = dt_f
    elif spk in tt_spks:
        f = tt_f
    else:
        f = tr_f

    for fname in fnames:
        if fname.endswith(".wav") or fname.endswith(".WAV"):
            sph_path = f"{root}/{fname}"
            path = f"{wav_dir}/{spk}_{fname}"
            uttid = f"{spk}_{os.path.splitext(fname)[0]}"
            f.write(f"{uttid} {path}\n")
            sph = SPHFile(sph_path)
            sph.write_wav(path)

tr_f.close()
dt_f.close()
tt_f.close()

print("converted to wav and dumped scp files")
