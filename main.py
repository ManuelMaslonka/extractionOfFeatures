import os
import pandas as pd

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
        results.append(features)

# Export
df = pd.DataFrame(results)
df.to_csv('extrahovane_features.csv', index=False)
print("✅ Všetko hotovo")


# Načítaj dáta
df = pd.read_csv('extrahovane_features.csv')

# Ensure numeric columns are properly converted and handle missing or invalid data
numeric_columns = ['zero_crossing_rate', 'spectral_flatness', 'tempo', 'onset_strength_mean',
                   'spectral_contrast_1', 'chroma_cqt_1', 'spectral_rolloff']

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, set invalid entries to NaN
    df[col] = df[col].fillna(0)  # Replace NaN with default value (0)


def classify_audio(row):
    # Extract all relevant features
    zcr = row.get('zero_crossing_rate', 0)
    flat = row.get('spectral_flatness', 0)
    tempo = row.get('tempo', 0)
    onset = row.get('onset_strength_mean', 0)
    chroma_cqt = row.get('chroma_cqt_1', 0)
    mfcc1 = row.get('mfcc_1', 0)
    contrast = row.get('spectral_contrast_1', 0)
    tonnetz = row.get('tonnetz_1', 0)
    rolloff = row.get('spectral_rolloff', 0)
    rms = row.get('rms', 0)
    bandwidth = row.get('spectral_bandwidth', 0)
    hp_ratio = row.get('harmonic_percussive_ratio', 0)
    centroid = row.get('spectral_centroid', 0)

    # === ŠUM (NOISE) ===
    if flat > 0.3:
        if zcr > 0.15 and centroid > 4000:
            return 'šum:biely' # White noise (high frequency content)
        elif centroid < 2000 and rolloff < 3000:
            return 'šum:ružový' # Pink noise (more low frequency energy)
        elif onset > 0.05 and bandwidth > 2000:
            return 'šum:ambient' # Ambient/environmental noise
        else:
            return 'šum:všeobecný' # General noise

    # === REČ (SPEECH) ===
    if flat < 0.2 and tempo < 180 and chroma_cqt < 0.25:
        if centroid < 1800 and mfcc1 < -50:
            return 'reč:muž' # Male speech (lower frequencies)
        elif centroid > 1800 and mfcc1 > -50:
            return 'reč:žena' # Female speech (higher frequencies)
        elif onset > 0.1 and contrast > 15:
            return 'reč:emotívna' # Emotional/excited speech
        elif onset < 0.05 and contrast < 10:
            return 'reč:monotónna' # Monotone/calm speech
        else:
            return 'reč:všeobecná' # General speech

    # === HUDBA (MUSIC) ===
    if tempo >= 60 or chroma_cqt >= 0.2 or tonnetz > 0.05:
        # Classical music tends to have higher harmonic content
        if hp_ratio > 1.5 and contrast > 30 and tempo < 120:
            return 'hudba:klasická'

        # Electronic music often has consistent beats and synthetic sounds
        elif flat > 0.15 and tempo > 120 and onset > 0.1:
            return 'hudba:elektronická'

        # Rock/pop typically has strong contrast and mid-range tempos
        elif contrast > 25 and 90 < tempo < 140 and onset > 0.08:
            return 'hudba:rock/pop'

        # Hip-hop often has strong bass and moderate tempo
        elif rolloff < 5000 and 70 < tempo < 110 and onset > 0.12:
            return 'hudba:hip-hop'

        # Ambient music has less pronounced beats and smoother spectra
        elif onset < 0.06 and contrast < 20 and flat < 0.1:
            return 'hudba:ambient'

        else:
            return 'hudba:všeobecná'

    # === SPECIAL CASES ===
    # Instrumental solo (high harmonic content, clear tonal structure)
    if hp_ratio > 2 and chroma_cqt > 0.3 and flat < 0.1:
        return 'hudba:sólový nástroj'

    # Percussion/drums (high percussive content, less harmonic structure)
    if hp_ratio < 0.5 and onset > 0.15 and chroma_cqt < 0.15:
        return 'hudba:perkusie'

    return 'nezaradené'


# Použi vylepšenú klasifikáciu
df['predikcia'] = df.apply(classify_audio, axis=1)

# Pridaj stĺpce pre hlavnú kategóriu a podkategóriu
df['hlavná_kategória'] = df['predikcia'].apply(lambda x: x.split(':')[0] if ':' in x else x)
df['podkategória'] = df['predikcia'].apply(lambda x: x.split(':')[1] if ':' in x else '')

# Export výsledku s detailnými kategóriami
df[['filename', 'predikcia', 'hlavná_kategória', 'podkategória']].to_csv('predikovane_typy_detailne.csv', index=False)
print("✅ Podrobné predikcie uložené do 'predikovane_typy_detailne.csv'")
