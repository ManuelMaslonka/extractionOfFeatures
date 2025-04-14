from extractors.base import FeatureExtractor
import librosa

class SpectralFlatnessFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the spectral flatness feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr):
        """
        Extracts the spectral flatness from the audio signal.
        Spectral flatness measures how noise-like a sound is.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the spectral flatness.
        """
        # Use librosa's built-in frame processing with our parameters
        flatness = librosa.feature.spectral_flatness(
            y=y, 
            n_fft=self.frame_length, 
            hop_length=self.hop_length
        ).mean()

        return {'spectral_flatness': flatness}
