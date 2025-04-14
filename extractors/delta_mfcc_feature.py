from extractors.base import FeatureExtractor
import librosa

class DeltaMFCCFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512, n_mfcc=13):
        """
        Initialize the delta MFCC feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
            n_mfcc (int): Number of MFCC coefficients to extract.
        """
        super().__init__(frame_length, hop_length)
        self.n_mfcc = n_mfcc

    def extract(self, y, sr):
        """
        Extract delta MFCC features from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the delta MFCC features.
        """
        # Use librosa's built-in frame processing with our parameters
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, 
                                   hop_length=self.hop_length, 
                                   n_fft=self.frame_length)

        # Compute delta (first derivative) of MFCC
        delta = librosa.feature.delta(mfcc)

        # Return the mean of each delta MFCC coefficient
        return {f'delta_mfcc_{i+1}': val for i, val in enumerate(delta.mean(axis=1))}
