from extractors.base import FeatureExtractor
import librosa

class ChromaCQTFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the chroma CQT feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr):
        """
        Extract chroma CQT features from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the chroma CQT features.
        """
        # Use librosa's built-in frame processing with our parameters
        chroma = librosa.feature.chroma_cqt(
            y=y, 
            sr=sr, 
            hop_length=self.hop_length
        )

        # Return the mean of each chroma bin
        return {f'chroma_cqt_{i+1}': val for i, val in enumerate(chroma.mean(axis=1))}
