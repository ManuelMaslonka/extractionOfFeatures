from extractors.base import FeatureExtractor
import librosa

class SpectralFlatnessFeature(FeatureExtractor):
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
        flatness = librosa.feature.spectral_flatness(y=y).mean()
        return {'spectral_flatness': flatness}
