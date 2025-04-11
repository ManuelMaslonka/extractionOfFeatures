from extractors.base import FeatureExtractor
import librosa
import numpy as np

class HarmonicPercussiveRatioFeature(FeatureExtractor):
    def extract(self, y, sr):
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_energy = np.sum(np.square(y_harm))
        perc_energy = np.sum(np.square(y_perc))
        ratio = harm_energy / (perc_energy + 1e-6)
        return {'harmonic_percussive_ratio': ratio}
