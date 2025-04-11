from extractors.base import FeatureExtractor
import librosa

class RMSFeature(FeatureExtractor):
    def extract(self, y, sr):
        """
        Extracts the RMS (Root Mean Square) energy from the audio signal.
        
        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.
            
        Returns:
            dict: Dictionary containing the RMS energy.
        """
        rms = librosa.feature.rms(y=y).mean()
        return {'rms': rms}
