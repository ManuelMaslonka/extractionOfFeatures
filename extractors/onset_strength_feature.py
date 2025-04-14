from extractors.base import FeatureExtractor
import librosa

class OnsetStrengthFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the onset strength feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr):
        """
        Extract onset strength features from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the onset strength features.
        """
        # Use librosa's built-in frame processing with our parameters
        onset_env = librosa.onset.onset_strength(
            y=y, 
            sr=sr, 
            hop_length=self.hop_length
        )

        # Return the mean and standard deviation of the onset strength envelope
        return {'onset_strength_mean': onset_env.mean(), 'onset_strength_std': onset_env.std()}
