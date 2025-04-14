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

# # Use the scaled features for classification
# df_for_classification = scaled_df.copy()
#
# def classify_audio(row):
#     zcr = row.get('zero_crossing_rate', 0)
#     flat = row.get('spectral_flatness', 0)
#     tempo = row.get('tempo', 0)
#     onset = row.get('onset_strength_mean', 0)
#     chroma_cqt = row.get('chroma_cqt_1', 0)
#     mfcc1 = row.get('mfcc_1', 0)
#     contrast = row.get('spectral_contrast_1', 0)
#     tonnetz = row.get('tonnetz_1', 0)
#     rolloff = row.get('spectral_rolloff', 0)
#     rms = row.get('rms', 0)
#     bandwidth = row.get('spectral_bandwidth', 0)
#     hp_ratio = row.get('harmonic_percussive_ratio', 0)
#     centroid = row.get('spectral_centroid', 0)
#
#     if flat > 0.3:
#         if zcr > 0.15 and centroid > 4000:
#             return 'šum:biely'
#         elif centroid < 2000 and rolloff < 3000:
#             return 'šum:ružový'
#         elif onset > 0.05 and bandwidth > 2000:
#             return 'šum:ambient'
#         else:
#             return 'šum:všeobecný'
#
#     if flat < 0.2 and tempo < 180 and chroma_cqt < 0.25:
#         if centroid < 1800 and mfcc1 < -50:
#             return 'reč:muž'
#         elif centroid > 1800 and mfcc1 > -50:
#             return 'reč:žena'
#         elif onset > 0.1 and contrast > 15:
#             return 'reč:emotívna'
#         elif onset < 0.05 and contrast < 10:
#             return 'reč:monotónna'
#         else:
#             return 'reč:všeobecná'
#
#     if tempo >= 60 or chroma_cqt >= 0.2 or tonnetz > 0.05:
#         if hp_ratio > 1.5 and contrast > 30 and tempo < 120:
#             return 'hudba:klasická'
#         elif flat > 0.15 and tempo > 120 and onset > 0.1:
#             return 'hudba:elektronická'
#         elif contrast > 25 and 90 < tempo < 140 and onset > 0.08:
#             return 'hudba:rock/pop'
#         elif rolloff < 5000 and 70 < tempo < 110 and onset > 0.12:
#             return 'hudba:hip-hop'
#         elif onset < 0.06 and contrast < 20 and flat < 0.1:
#             return 'hudba:ambient'
#         else:
#             return 'hudba:všeobecná'
#
#     if hp_ratio > 2 and chroma_cqt > 0.3 and flat < 0.1:
#         return 'hudba:sólový nástroj'
#     if hp_ratio < 0.5 and onset > 0.15 and chroma_cqt < 0.15:
#         return 'hudba:perkusie'
#
#     return 'nezaradené'
#
# df_for_classification['predikcia'] = df_for_classification.apply(classify_audio, axis=1)
# df_for_classification['hlavná_kategória'] = df_for_classification['predikcia'].apply(lambda x: x.split(':')[0] if ':' in x else x)
# df_for_classification['podkategória'] = df_for_classification['predikcia'].apply(lambda x: x.split(':')[1] if ':' in x else '')
#
# df_for_classification[['filename', 'label', 'predikcia', 'hlavná_kategória', 'podkategória']].to_csv('predikovane_typy_detailne.csv', index=False)
# print("✅ Podrobné predikcie uložené do 'predikovane_typy_detailne.csv'")
