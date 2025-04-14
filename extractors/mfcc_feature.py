from extractors.base import FeatureExtractor
import librosa
import numpy as np

class MFCCFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512, n_mfcc=13):
        """
        Initialize the MFCC feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
            n_mfcc (int): Number of MFCC coefficients to extract.
        """
        super().__init__(frame_length, hop_length)
        self.n_mfcc = n_mfcc

    def extract(self, y, sr):
        """
        Extract MFCC features from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the MFCC features.
        """
        # Use librosa's built-in frame processing with our parameters
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, 
                                     hop_length=self.hop_length, 
                                     n_fft=self.frame_length)

        # Return the mean of each MFCC coefficient
        return {f'mfcc_{i+1}': coef for i, coef in enumerate(mfccs.mean(axis=1))}
