from extractors.base import FeatureExtractor
import librosa

class SpectralContrastFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the spectral contrast feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr):
        """
        Extract spectral contrast features from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the spectral contrast features.
        """
        # Use librosa's built-in frame processing with our parameters
        contrast = librosa.feature.spectral_contrast(
            y=y, 
            sr=sr, 
            n_fft=self.frame_length, 
            hop_length=self.hop_length
        )

        # Return the mean of each contrast band
        return {f'spectral_contrast_{i+1}': val for i, val in enumerate(contrast.mean(axis=1))}
