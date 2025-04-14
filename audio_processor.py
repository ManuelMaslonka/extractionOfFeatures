import os
import librosa

class AudioProcessor:
    def __init__(self, extractors, frame_length=2048, hop_length=512):
        """
        Initialize the audio processor with a list of feature extractors and frame parameters.

        Parameters:
            extractors (list): List of feature extractors.
            frame_length (int): Length of the frame in samples. Default is 2048 samples (about 46ms at 44.1kHz).
            hop_length (int): Number of samples between successive frames. Default is 512 samples (about 11.6ms at 44.1kHz).
        """
        self.extractors = extractors
        self.frame_length = frame_length
        self.hop_length = hop_length

        # Set frame_length and hop_length for each extractor
        for extractor in self.extractors:
            extractor.frame_length = self.frame_length
            extractor.hop_length = self.hop_length

    def process(self, file_path):
        """
        Process an audio file and extract features.

        Parameters:
            file_path (str): Path to the audio file.

        Returns:
            dict: Dictionary containing the extracted features or None if the file couldn't be processed.
        """
        try:
            y, sr = librosa.load(file_path, sr=None)
            features = {'filename': os.path.basename(file_path)}

            for extractor in self.extractors:
                features.update(extractor.extract(y, sr))
            return features
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
            return None
