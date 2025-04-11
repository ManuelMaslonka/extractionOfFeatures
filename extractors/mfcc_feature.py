from extractors.base import FeatureExtractor
import librosa

class MFCCFeature(FeatureExtractor):
    def extract(self, y, sr):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return {f'mfcc_{i+1}': coef for i, coef in enumerate(mfccs.mean(axis=1))}
