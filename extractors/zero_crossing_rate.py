import librosa

from extractors.base import FeatureExtractor


class ZeroCrossingRateExtractor(FeatureExtractor):
    def extract(self, y, sr) -> dict:
        """
        Extracts the zero-crossing rate from the audio signal.

        Parameters:
            y (ndarray): Audio time series.
            sr (int): Sampling rate of `y`.

        Returns:
            dict: Dictionary containing the zero-crossing rate.
        """
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        return {'zero_crossing_rate': zero_crossing_rate}
