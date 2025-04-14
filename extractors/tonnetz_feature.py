from extractors.base import FeatureExtractor
import librosa

class TonnetzFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the tonnetz feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr):
        """
        Extract tonnetz features from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the tonnetz features.
        """
        # Extract the harmonic component of the signal
        y_harmonic = librosa.effects.harmonic(y)

        # Use librosa's built-in frame processing with our parameters
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr, 
                                         hop_length=self.hop_length)

        # Return the mean of each tonnetz dimension
        return {f'tonnetz_{i+1}': val for i, val in enumerate(tonnetz.mean(axis=1))}
