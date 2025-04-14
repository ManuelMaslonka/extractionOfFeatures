from extractors.base import FeatureExtractor
import librosa

class SpectralBandwidthFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the spectral bandwidth feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr):
        """
        Extracts the spectral bandwidth from the audio signal.
        Spectral bandwidth represents the variance of the spectrum around its centroid.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the spectral bandwidth.
        """
        # Use librosa's built-in frame processing with our parameters
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, 
            sr=sr, 
            n_fft=self.frame_length, 
            hop_length=self.hop_length
        ).mean()

        return {'spectral_bandwidth': bandwidth}
