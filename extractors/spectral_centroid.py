from extractors.base import FeatureExtractor
import librosa

class SpectralCentroidFeature(FeatureExtractor):
    def extract(self, y, sr):
        """
        Extracts the spectral centroid from the audio signal.
        """
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        return {'spectral_centroid': centroid}
