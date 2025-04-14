from abc import ABC, abstractmethod
import numpy as np
import librosa

class FeatureExtractor(ABC):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the feature extractor with frame and hop length parameters.

        Parameters:
            frame_length (int): Length of the frame in samples. Default is 2048 samples (about 46ms at 44.1kHz).
            hop_length (int): Number of samples between successive frames. Default is 512 samples (about 11.6ms at 44.1kHz).
        """
        self.frame_length = frame_length
        self.hop_length = hop_length

    def segment_signal(self, y):
        """
        Segment the audio signal into frames.

        Parameters:
            y (ndarray): Audio time series.

        Returns:
            ndarray: Segmented frames of the audio signal.
        """
        return librosa.util.frame(y, frame_length=self.frame_length, hop_length=self.hop_length)

    def apply_window(self, frames):
        """
        Apply a Hamming window to each frame.

        Parameters:
            frames (ndarray): Segmented frames of the audio signal.

        Returns:
            ndarray: Windowed frames.
        """
        window = np.hamming(self.frame_length)
        return frames * window[:, np.newaxis]

    @abstractmethod
    def extract(self, y, sr) -> dict:
        """
        Extract features from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the extracted features.
        """
        pass
