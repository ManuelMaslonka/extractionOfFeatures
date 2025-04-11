from extractors.base import FeatureExtractor
import librosa

class TempoFeature(FeatureExtractor):
    def extract(self, y, sr):
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return {'tempo': tempo}
