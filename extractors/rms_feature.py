from extractors.base import FeatureExtractor
import librosa

class RMSFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the RMS feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr):
        """
        Extracts the RMS (Root Mean Square) energy from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the RMS energy.
        """
        # Use librosa's built-in frame processing with our parameters
        rms = librosa.feature.rms(y=y, 
                                 frame_length=self.frame_length, 
                                 hop_length=self.hop_length).mean()
        return {'rms': rms}
