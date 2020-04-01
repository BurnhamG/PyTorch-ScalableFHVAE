import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import os
from collections import OrderedDict
import numpy as np
import json
from kaldiio import load_scp
from pathlib import Path


def scp2dict(path, dtype=str, seqlist=None):
    """Convert an scp file to a dictionary.

    Args:
        path:    Path to scp file
        dtype:   Data type the dictionary value should be cast to
        seqlist: If not None, limits returned dictionary to the keys in this list

    Returns:
        An OrderedDict with the keys and values from the first and second columns
            of the scp file, respectively

    """
    with open(path) as f:
        line_list = [line.rstrip().split(None, 1) for line in f]
    if seqlist is None:
        d = OrderedDict([(k, dtype(v)) for k, v in line_list])
    else:
        d = OrderedDict([(k, dtype(v)) for k, v in line_list if k in seqlist])
    return d


class Segment(object):
    """Represents an audio segment."""

    def __init__(self, seq, start, end):
        self.seq = seq
        self.start = start
        self.end = end

    def __str__(self):
        return f"{self.seq}, {self.start}, {self.end}"

    def __repr__(self):
        return str(self)


class BaseDataset(Dataset):
    def __init__(
        self,
        feat_scp: Path,
        len_scp: Path,
        min_len: int = 1,
        mvn_path: str = None,
        seg_len: int = 20,
        seg_shift: int = 8,
        rand_seg: bool = False,
    ):
        """
        Args:
            feat_scp:  Feature scp path
            len_scp:   Sequence-length scp path
            min_len:   Keep sequence no shorter than min_len
            mvn_path:       Path to file storing the mean and variance of the sequences
                                for normalization
            seg_len:   Segment length
            seg_shift: Segment shift if seg_rand is False; otherwise randomly
                                extract floor(seq_len/seg_shift) segments per sequence
            rand_seg: If True, randomly extract segments
        """
        feats = scp2dict(feat_scp)
        lens = scp2dict(len_scp, int, feats.keys())

        self.seg_len = seg_len
        self.seg_shift = seg_shift
        self.rand_seg = rand_seg

        self.seqlist = [k for k in feats.keys() if lens[k] >= min_len]
        self.feats = OrderedDict([(k, feats[k]) for k in self.seqlist])
        self.lens = OrderedDict([(k, lens[k]) for k in self.seqlist])
        print(
            f"{self.__class__.__name__}: {len(self.feats)} out of {len(feats)} kept, min_len = {min_len}"
        )

        self.seq_keys, self.seq_feats, self.seq_lens = self._make_seq_lists(
            self.seqlist
        )
        self.segs, self.seq_nsegs = self._make_segs(
            self.seq_keys, self.seq_lens, self.seg_len, self.seg_shift, self.rand_seg
        )
        self.seq2idx = dict([(seq, i) for i, seq in enumerate(self.seq_keys)])

        if mvn_path is not None:
            if not os.path.exists(mvn_path):
                self.mvn_params = self.compute_mvn()
                with open(mvn_path, "w") as f:
                    json.dump(self.mvn_params, f)
            else:
                with open(mvn_path) as f:
                    self.mvn_params = json.load(f)
        else:
            self.mvn_params = None

    def apply_mvn(self, feats):
        """Apply mean and variance normalization."""
        if self.mvn_params is None:
            return feats
        else:
            return (feats - self.mvn_params["mean"]) / self.mvn_params["std"]

    def compute_mvn(self):
        """Compute mean and variance normalization."""
        n, x, x2 = 0.0, 0.0, 0.0
        for seq in self.seqlist:
            feat = self.feats[seq]
            x += np.sum(feat, axis=0, keepdims=True)
            x2 += np.sum(feat ** 2, axis=0, keepdims=True)
            n += feat.shape[0]
        mean = x / n
        std = np.sqrt(x2 / n - mean ** 2)
        return {"mean": mean, "std": std}

    def undo_mvn(self, feats):
        """Undo mean and variance normalization."""
        if self.mvn_params is None:
            return feats
        else:
            return feats * self.mvn_params["std"] + self.mvn_params["mean"]

    def __len__(self):
        return len(self.seqlist)

    def __getitem__(self, index):
        """Returns key(sequence), feature, and number of segments."""
        seg = self.segs[index]
        idx = self.seq2idx[seg.seq]
        key = self.seq_keys[idx]
        feat = self.seq_feats[idx][seg.start : seg.end]
        nsegs = self.seq_nsegs[idx]

        return key, feat, nsegs

    def _make_seq_lists(self, seqlist):
        """Return lists of all sequences and the corresponding features and lengths."""
        keys, feats, lens = [], [], []
        for seq in seqlist:
            keys.append(seq)
            feats.append(self.feats[seq])
            lens.append(self.lens[seq])

        return keys, feats, lens

    def _make_segs(
        self,
        seqs: list,
        lens: list,
        seg_len: int = 20,
        seg_shift: int = 8,
        rand_seg: bool = False,
    ):
        """Make segments from a list of sequences.

        Args:
            seqs:      List of sequences
            lens:      List of sequence lengths
            seg_len:   Segment length
            seg_shift: Segment shift if rand_seg is False; otherwise randomly
                           extract floor(seq_len/seg_shift) segments per sequence
            rand_seg:  If True, randomly extract segments
        """
        segs = []
        nsegs = []
        for seq, l in zip(seqs, lens):
            nseg = (l - seg_len) // seg_shift + 1
            nsegs.append(nseg)
            if rand_seg:
                starts = np.random.choice(range(l - seg_len + 1), nseg)
            else:
                starts = np.arange(nseg) * seg_shift
            for start in starts:
                end = start + seg_len
                segs.append(Segment(seq, start, end))
        return segs, nsegs


class NumpyDataset(BaseDataset):
    def __init__(
        self,
        feat_scp: Path,
        len_scp: Path,
        min_len: int = 1,
        mvn_path: str = None,
        seg_len: int = 20,
        seg_shift: int = 8,
        rand_seg: bool = False,
    ):
        """
        Args:
            feat_scp:  Feature scp path
            len_scp:   Sequence-length scp path
            min_len:   Keep sequence no shorter than min_len
            seg_len:   Segment length
            seg_shift: Segment shift if seg_rand is False; otherwise randomly
                                extract floor(seq_len/seg_shift) segments per sequence
            rand_seg: If True, randomly extract segments
        """
        super().__init__(
            feat_scp, len_scp, min_len, mvn_path, seg_len, seg_shift, rand_seg
        )

        self.feats = self.feat_getter(self.feats)

    class feat_getter:
        def __init__(self, feats):
            self.feats = dict(feats)

        def __getitem__(self, seq):
            with open(self.feats[seq]) as f:
                feat = np.load(f)
            return feat

    def __getitem__(self, index):
        """Returns key(sequence), feature, and number of segments."""
        seg = self.segs[index]
        idx = self.seq2idx[seg.seq]
        key = self.seq_keys[idx]
        with open(self.seq_feats[idx]) as f:
            feat = np.load(f)[seg.start : seg.end]
        nsegs = self.seq_nsegs[idx]

        return key, feat, nsegs


class KaldiDataset(BaseDataset):
    def __init__(
        self,
        feat_scp: Path,
        len_scp: Path,
        min_len: int = 1,
        mvn_path: str = None,
        seg_len: int = 20,
        seg_shift: int = 8,
        rand_seg: bool = False,
    ):
        super().__init__(
            feat_scp, len_scp, min_len, mvn_path, seg_len, seg_shift, rand_seg
        )
        self.feats = load_scp(f"scp:{feat_scp}")

    def __getitem__(self, index):
        """Returns key(sequence), feature, and number of segments."""
        seg = self.segs[index]
        idx = self.seq2idx[seg.seq]
        key = self.seq_keys[idx]
        feat = self.seq_feats[idx][seg.start : seg.end]
        nsegs = self.seq_nsegs[idx]

        return key, feat, nsegs
