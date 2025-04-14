from extractors.base import FeatureExtractor
import librosa

class TempogramFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the tempogram feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr):
        """
        Extract tempogram features from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the tempogram features.
        """
        # Compute onset strength envelope with our parameters
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, 
                                                hop_length=self.hop_length)

        # Compute tempogram with our parameters
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, 
                                             hop_length=self.hop_length)

        # Return the mean of each tempogram dimension
        return {f'tempogram_{i+1}': val for i, val in enumerate(tempogram.mean(axis=1))}
