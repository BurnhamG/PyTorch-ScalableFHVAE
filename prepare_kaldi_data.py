import os
import argparse
import subprocess


def prepare_kaldi(
    dataset,
    wav_scp,
    feat_ark,
    feat_scp,
    len_scp,
    fbank_conf="./misc/fbank.conf",
    kaldi_root="./kaldi",
) -> None:
    for p in [feat_ark, feat_scp, len_scp]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    feat_comp_cmd = [
        os.path.join(kaldi_root, "src/bin/compute-fbank-feats"),
        f"scp,p:{wav_scp}",
        f"ark,scp:{feat_ark},{feat_scp}",
        f"--config={fbank_conf}",
    ]
    feat_compute = subprocess.Popen(feat_comp_cmd)

    feat_len_cmd = [
        os.path.join(kaldi_root, "src/featbin/feat-to-len"),
        f"scp:{feat_scp}",
        f"ark,t:{len_scp}",
    ]
    feat_to_len = subprocess.Popen(feat_len_cmd)

    exit_codes = [p.wait() for p in (feat_compute, feat_to_len)]
    if any(exit_codes):
        commands = (" ".join(feat_comp_cmd), " ".join(feat_len_cmd))
        error_info = [
            f"{cmd}: {error}" for (cmd, error) in zip(commands, exit_codes) if error > 0
        ]
        raise RuntimeError(f"Non-zero return code(s): {', '.join(error_info)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("wav_scp", type=str, help="Input wav scp file")
    parser.add_argument("feat_ark", type=str, help="Output feats.ark file")
    parser.add_argument("feat_scp", type=str, help="Output feats.scp file")
    parser.add_argument("len_scp", type=str, help="Output len.scp file")
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech",
        choices=["librispeech", "timit"],
        help="Input wav scp file",
    )
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
        args.dataset,
        args.wav_scp,
        args.feat_ark,
        args.feat_scp,
        args.len_scp,
        args.fbank_conf,
        args.kaldi_root,
    )
