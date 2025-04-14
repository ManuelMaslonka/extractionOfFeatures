from extractors.base import FeatureExtractor
import librosa
import numpy as np

class HarmonicPercussiveRatioFeature(FeatureExtractor):
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the harmonic percussive ratio feature extractor.

        Parameters:
            frame_length (int): Length of the frame in samples.
            hop_length (int): Number of samples between successive frames.
        """
        super().__init__(frame_length, hop_length)

    def extract(self, y, sr):
        """
        Extract harmonic percussive ratio feature from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the harmonic percussive ratio feature.
        """
        # Separate harmonic and percussive components
        y_harm, y_perc = librosa.effects.hpss(y, margin=2.0)

        # Compute energy of each component
        harm_energy = np.sum(np.square(y_harm))
        perc_energy = np.sum(np.square(y_perc))

        # Compute ratio (avoid division by zero)
        ratio = harm_energy / (perc_energy + 1e-6)

        return {'harmonic_percussive_ratio': ratio}
