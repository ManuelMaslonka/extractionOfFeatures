from extractors.base import FeatureExtractor
import librosa

class SpectralRolloffFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the spectral rolloff feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr):
        """
        Extract spectral rolloff feature from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the spectral rolloff feature.
        """
        # Use librosa's built-in frame processing with our parameters
        rolloff = librosa.feature.spectral_rolloff(
            y=y, 
            sr=sr, 
            n_fft=self.frame_length, 
            hop_length=self.hop_length
        ).mean()

        return {'spectral_rolloff': rolloff}
