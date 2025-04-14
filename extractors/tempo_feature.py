from extractors.base import FeatureExtractor
import librosa

class TempoFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the tempo feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr):
        """
        Extract tempo feature from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the tempo feature.
        """
        # Use librosa's built-in frame processing with our parameters
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        return {'tempo': tempo}
