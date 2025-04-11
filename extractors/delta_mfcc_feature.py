from extractors.base import FeatureExtractor
import librosa

class DeltaMFCCFeature(FeatureExtractor):
    def extract(self, y, sr):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        return {f'delta_mfcc_{i+1}': val for i, val in enumerate(delta.mean(axis=1))}
