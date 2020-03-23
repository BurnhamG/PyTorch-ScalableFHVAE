import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import os
from collections import OrderedDict
import numpy as np


def scp2dict(path, dtype=str, seqlist=None):
    """Convert an scp file to a dictionary

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


def make_segs(seqs, lens, labs, talabs, seg_len, seg_shift, rand_seg):
    """
    Args:
        seqs(list): list of sequences
        lens(list): list of sequence lengths
        labs(list): list of sequence label lists
        talabs(list): list of sequence time-aligned label sequence lists
        seg_len(int):
        seg_shift(int):
        rand_seg(bool):
    """
    segs = []
    nsegs = []
    for seq, l, lab, talab in zip(seqs, lens, labs, talabs):
        nseg = (l - seg_len) // seg_shift + 1
        nsegs.append(nseg)
        if rand_seg:
            starts = np.random.choice(range(l - seg_len + 1), nseg)
        else:
            starts = np.arange(nseg) * seg_shift
        for start in starts:
            end = start + seg_len
            seg_talab = [s.center_lab(start, end) for s in talab]
            segs.append(Segment(seq, start, end, lab, seg_talab))
    return segs, nsegs


class Segment(object):
    def __init__(self, seq, start, end, lab, talab):
        self.seq = seq
        self.start = start
        self.end = end
        self.lab = lab
        self.talab = talab

    def __str__(self):
        return f"{self.seq}, {self.start}, {self.end}, {self.lab},{ self.talab}"

    def __repr__(self):
        return str(self)


class NumpyDataset(Dataset):
    def __init__(
        self,
        feat_scp: str,
        len_scp: str,
        lab_specs: list = None,
        talab_specs: list = None,
        min_len: int = 1,
    ):
        """
        Args:
            feat_scp(str): feature scp path
            len_scp(str): sequence-length scp path
            lab_specs(list): list of label specifications. each is
                (name, number of classes, scp path)
            talab_specs(list): list of time-aligned label specifications.
                each is (name, number of classes, ali path)
            min_len(int): keep sequence no shorter than min_len
        """
        feats = scp2dict(feat_scp)
        lens = scp2dict(len_scp, int, feats.keys())

        self.seqlist = [k for k in feats.keys() if lens[k] >= min_len]
        self.feats = OrderedDict([(k, feats[k]) for k in self.seqlist])
        self.lens = OrderedDict([(k, lens[k]) for k in self.seqlist])

        self.labs_d = OrderedDict()
        self.talabseqs_d = OrderedDict()
        if lab_specs is not None:
            for lab_spec in lab_specs:
                name, nclass, seq2lab = load_lab(lab_spec, self.seqlist)
                self.labs_d[name] = Labels(name, nclass, seq2lab)
        if talab_specs is not None:
            for talab_spec in talab_specs:
                name, nclass, seq2talabs = load_talab(talab_spec, self.seqlist)
                self.talabseqs_d[name] = TimeAlignedLabelSeqs(name, nclass, seq2talabs)

    def __len__(self):
        return len(self.seqlist)

    def __getitem__(self, index):
        """Returns key, feat, sequence length, included label, included time-aligned label"""
        return self.seqlist[index], self.feats[index], self.lens[index]

    def _make_seqs(self, seqlist, lab_names, talab_names, shuffle=False):
        """
        Yields:
            key:   Sequence, used as a key
            feat:  Feature for the corresponding sequence
            leng:  Length of the corresponding sequence
            lab:   List of corresponding labels(int list)
            talab: List of corresponding time-aligned labels(talabel list)
        """
        if shuffle:
            np.random.shuffle(seqlist)

        lab, talab = [], []
        for seq in seqlist:
            key = seq
            feat = self.feats[seq]
            leng = self.lens[seq]
            lab = [self.labs_d[name][seq] for name in lab_names]
            talab = [self.talabseqs_d[name][seq] for name in talab_names]
            yield key, feat, leng, lab, talab


# class KaldiDataset(Dataset):
