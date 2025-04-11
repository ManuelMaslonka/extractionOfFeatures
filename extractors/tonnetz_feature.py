from extractors.base import FeatureExtractor
import librosa

class TonnetzFeature(FeatureExtractor):
    def extract(self, y, sr):
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        return {f'tonnetz_{i+1}': val for i, val in enumerate(tonnetz.mean(axis=1))}
