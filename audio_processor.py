import os
import librosa

class AudioProcessor:
    def __init__(self, extractors):
        self.extractors = extractors

    def process(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        features = {'filename': os.path.basename(file_path)}

        for extractor in self.extractors:
            features.update(extractor.extract(y, sr))
        return features
