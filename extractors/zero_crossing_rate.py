import librosa

from extractors.base import FeatureExtractor


class ZeroCrossingRateExtractor(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the zero crossing rate feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr) -> dict:
        """
        Extracts the zero-crossing rate from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the zero-crossing rate.
        """
        # Use librosa's built-in frame processing with our parameters
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y, 
            frame_length=self.frame_length, 
            hop_length=self.hop_length
        ).mean()

        return {'zero_crossing_rate': zero_crossing_rate}
