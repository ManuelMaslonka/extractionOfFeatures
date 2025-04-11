from extractors.base import FeatureExtractor
import librosa

class ChromaCENSFeature(FeatureExtractor):
    def extract(self, y, sr):
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        return {f'chroma_cens_{i+1}': val for i, val in enumerate(chroma.mean(axis=1))}
