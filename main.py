import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from audio_processor import AudioProcessor

# Import all feature extractors
from extractors.rms_feature import RMSFeature
from extractors.mfcc_feature import MFCCFeature
from extractors.chroma_feature import ChromaFeature
from extractors.tonnetz_feature import TonnetzFeature
from extractors.zero_crossing_rate import ZeroCrossingRateExtractor
from extractors.spectral_centroid import SpectralCentroidFeature
from extractors.spectral_bandwidth import SpectralBandwidthFeature
from extractors.spectral_flatness import SpectralFlatnessFeature
from extractors.spectral_contrast import SpectralContrastFeature
from extractors.spectral_rolloff import SpectralRolloffFeature
from extractors.delta_mfcc_feature import DeltaMFCCFeature
from extractors.chroma_cqt_feature import ChromaCQTFeature
from extractors.chroma_cens_feature import ChromaCENSFeature
from extractors.tempo_feature import TempoFeature
from extractors.tempogram_feature import TempogramFeature
from extractors.onset_strength_feature import OnsetStrengthFeature
from extractors.harmonic_percussive_ratio_feature import HarmonicPercussiveRatioFeature

# Initialize extractors
extractors = [
    RMSFeature(),
    MFCCFeature(),
    ChromaFeature(),
    TonnetzFeature(),
    ZeroCrossingRateExtractor(),
    SpectralCentroidFeature(),
    SpectralBandwidthFeature(),
    SpectralFlatnessFeature(),
    SpectralContrastFeature(),
    SpectralRolloffFeature(),
    DeltaMFCCFeature(),
    ChromaCQTFeature(),
    ChromaCENSFeature(),
    TempoFeature(),
    TempogramFeature(),
    OnsetStrengthFeature(),
    HarmonicPercussiveRatioFeature()
]

processor = AudioProcessor(extractors)

# Process files

folder_path = 'hudba'
results = []

for file in os.listdir(folder_path):
    if file.lower().endswith(('.wav', '.mp3', '.flac')):
        print(f"🎧 Spracovávam: {file}")
        full_path = os.path.join(folder_path, file)
        features = processor.process(full_path)
        if features is not None:
            results.append(features)
        else:
            print(f"⚠️ Skipping {file} due to processing error")

# Export
df = pd.DataFrame(results)
df.to_csv('extrahovane_features.csv', index=False)
print("✅ Všetko hotovo")

# Načítaj dáta
df = pd.read_csv('extrahovane_features.csv')

# Identify all numeric columns (excluding 'filename' which is a string)
numeric_columns = df.columns.drop('filename')

# Convert all feature columns to numeric and handle missing values
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(0)

# Apply StandardScaler to normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[numeric_columns])

# Create a new DataFrame with the scaled features
scaled_df = pd.DataFrame(scaled_features, columns=numeric_columns)
scaled_df['filename'] = df['filename']

# 🔥 Vytiahni label (štýl) zo súborového názvu
def extract_label(filename):
    if '.' in filename:
        return filename.split('.')[0].lower()
    return 'nezname'

scaled_df['label'] = scaled_df['filename'].apply(extract_label)

# Save with label
scaled_df.to_csv('features_s_labelmi.csv', index=False)
print("✅ Features s labelmi uložené do 'features_s_labelmi.csv'")