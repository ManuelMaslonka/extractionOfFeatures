from extractors.base import FeatureExtractor
import librosa

class ChromaFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the chroma feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr):
        """
        Extract chroma features from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the chroma features.
        """
        # Use librosa's built-in frame processing with our parameters
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, 
                                            hop_length=self.hop_length, 
                                            n_fft=self.frame_length)

        # Return the mean of each chroma bin
        return {f'chroma_{i+1}': chroma.mean(axis=1)[i] for i in range(chroma.shape[0])}
