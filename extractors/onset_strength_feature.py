from extractors.base import FeatureExtractor
import librosa

class OnsetStrengthFeature(FeatureExtractor):
    def extract(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        return {'onset_strength_mean': onset_env.mean(), 'onset_strength_std': onset_env.std()}
