import librosa
import numpy as np
from nptyping import Array
from pathlib import Path


def create_output_dir(dataset: str, feat_type: str, data_format: str) -> Path:
    """Concatenates the dataset name, format, and feature type to create a dir name"""
    if data_format.lower() == "numpy":
        dataset += "_np"
    else:
        dataset += "_kd"

    # Kaldi only computes fbank features
    feat_type = "fbank" if data_format == "kaldi" else feat_type

    return Path(dataset + f"_{feat_type}")


class AudioUtils:
    @staticmethod
    def stft(
        y: Array[float],
        sr: int,
        n_fft: int = 400,
        hop_t: float = 0.010,
        win_t: float = 0.025,
        window: str = "hamming",
        preemphasis: float = 0.97,
    ) -> Array[float]:
        """Short time Fourier Transform

        Args:
            y:           Raw waveform of shape (T,)
            sr:          Sample rate
            n_fft:       Length of the FFT window
            hop_t:       Spacing (in seconds) between consecutive frames
            win_t:       Window size (in seconds)
            window:      Type of window applied for STFT
            preemphasis: Pre-emphasize raw signal with y[t] = x[t] - r*x[t-1]

        Returns:
            (n_fft / 2 + 1, N) matrix; N is number of frames

        """
        if preemphasis > 1e-12:
            y = y - preemphasis * np.concatenate([[0], y[:-1]], 0)
        hop_length = int(sr * hop_t)
        win_length = int(sr * win_t)
        return librosa.core.stft(
            y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
        )

    @staticmethod
    def rstft(
        y: Array[float],
        sr: int,
        n_fft: int = 400,
        hop_t: float = 0.010,
        win_t: float = 0.025,
        window: str = "hamming",
        preemphasis: float = 0.97,
        log: bool = True,
        log_floor: int = -50,
    ) -> Array[float]:
        """Computes (log) magnitude spectrogram

        Args:
            y:           Raw waveform of shape (T,)
            sr:          Sample rate
            n_fft:       Length of the FFT window
            hop_t:       Spacing (in seconds) between consecutive frames
            win_t:       Window size (in seconds)
            window:      Type of window applied for STFT
            preemphasis: Pre-emphasize raw signal with y[t] = x[t] - r*x[t-1]
            log:         If True, uses log magnitude
            log_floor:   Floor value for log scaling of magnitude

        Returns:
            (n_fft / 2 + 1, N) matrix; N is number of frames

        """
        spec = AudioUtils.stft(y, sr, n_fft, hop_t, win_t, window, preemphasis)
        spec = np.abs(spec)
        if log:
            spec = np.log(spec)
            spec[spec < log_floor] = log_floor
        return spec

    @staticmethod
    def to_melspec(
        y: Array[float, 1],
        sr: int,
        n_fft: int = 400,
        hop_t: float = 0.010,
        win_t: float = 0.025,
        window: str = "hamming",
        preemphasis: float = 0.97,
        n_mels: int = 80,
        log: bool = True,
        norm_mel: bool = False,
        log_floor: int = -20,
    ) -> Array[float]:
        """Compute Mel-scale filter bank coefficients:

        Args:
            y:           Numpy array of audio sample
            sr:          Sample rate
            hop_t:       Spacing (in seconds) between consecutive frames
            win_t:       Window size (in seconds)
            window:      Type of window applied for STFT
            preemphasis: Pre-emphasize raw signal with y[t] = x[t] - r*x[t-1]
            n_mels:      Number of filter banks, which are equally spaced in Mel-scale
            log:         If True, use log magnitude
            norm_mel:    Normalize each filter bank to have area of 1 if True;
                             otherwise the peak value of each filter bank is 1

        Returns:
            (n_mels, N) matrix; N is number of frames

        """
        spec = AudioUtils.rstft(
            y, sr, n_fft, hop_t, win_t, window, preemphasis, log=False
        )
        hop_length = int(sr * hop_t)
        melspec = librosa.feature.melspectrogram(
            sr=sr,
            S=spec,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            norm=norm_mel,
        )
        if log:
            melspec = np.log(melspec)
            melspec[melspec < log_floor] = log_floor
        return melspec

    @staticmethod
    def energy_vad(
        y: Array[float],
        sr: int,
        hop_t: float = 0.010,
        win_t: float = 0.025,
        th_ratio: float = 1.04 / 2,
    ) -> Array[bool]:
        """Compute energy-based VAD (Voice activity detection)

        Args:
            y:        Numpy array of audio sample
            sr:       Sample rate
            hop_t:    Spacing (in seconds) between consecutive frames
            win_t:    Window size (in seconds)
            th_ratio: Energy threshold ratio for detection

        Returns:
            Boolean vector indicating whether voice was detected

        """
        hop_length = int(sr * hop_t)
        win_length = int(sr * win_t)
        e = librosa.feature.rmse(y, frame_length=win_length, hop_length=hop_length)
        th = th_ratio * np.mean(e)
        vad = np.asarray(e > th, dtype=int)
        return vad
