from extractors.base import FeatureExtractor
import librosa

class TempogramFeature(FeatureExtractor):
    def extract(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        return {f'tempogram_{i+1}': val for i, val in enumerate(tempogram.mean(axis=1))}
