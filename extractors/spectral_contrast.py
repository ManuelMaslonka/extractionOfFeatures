from extractors.base import FeatureExtractor
import librosa

class SpectralContrastFeature(FeatureExtractor):
    def extract(self, y, sr):
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        return {f'spectral_contrast_{i+1}': val for i, val in enumerate(contrast.mean(axis=1))}
