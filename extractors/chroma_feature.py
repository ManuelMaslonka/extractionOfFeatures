from extractors.base import FeatureExtractor
import librosa

class ChromaFeature(FeatureExtractor):
    def extract(self, y, sr):
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        return {f'chroma_{i+1}': chroma.mean(axis=1)[i] for i in range(chroma.shape[0])}
