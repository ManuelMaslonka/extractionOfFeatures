from extractors.base import FeatureExtractor
import librosa

class SpectralRolloffFeature(FeatureExtractor):
    def extract(self, y, sr):
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        return {'spectral_rolloff': rolloff}
