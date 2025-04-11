from extractors.base import FeatureExtractor
import librosa

class SpectralBandwidthFeature(FeatureExtractor):
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
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        return {'spectral_bandwidth': bandwidth}
