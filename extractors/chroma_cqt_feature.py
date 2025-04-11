from extractors.base import FeatureExtractor
import librosa

class ChromaCQTFeature(FeatureExtractor):
    def extract(self, y, sr):
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        return {f'chroma_cqt_{i+1}': val for i, val in enumerate(chroma.mean(axis=1))}
