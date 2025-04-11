# Audio Classification Tool

### Overview
* This project is an audio classification system that analyzes audio files and categorizes them into three classes: noise ("šum"), speech ("reč"), and music ("hudba"). Built as a school project, it demonstrates the application of signal processing techniques and feature extraction for audio classification tasks.

### Features
- Extracts 17+ audio features from sound files using advanced DSP techniques
- Processes WAV, MP3, and FLAC audio formats
- Rule-based classification system based on acoustic properties
- Exports feature data and classification results to CSV files

### How It Works
The system follows a pipeline approach:

1. **Feature Extraction**: Processes audio files to extract features like MFCCs, chroma, spectral properties, etc.
2. **Feature Analysis**: Aggregates and normalizes the extracted features
3. **Classification**: Uses rule-based heuristics to categorize audio based on extracted features
4. **Results Export**: Saves both raw features and classification results to CSV files

### Extracted Audio Features
- Root Mean Square (RMS) energy
- Mel-Frequency Cepstral Coefficients (MFCCs)
- Chroma features (regular, CQT, CENS)
- Tonnetz (tonal centroid features)
- Zero-crossing rate
- Spectral properties (centroid, bandwidth, flatness, contrast, rolloff)
- Tempo and rhythm features
- Onset strength
- Harmonic/percussive ratio

### Requirements
- **Python 3.x**
- Libraries:
    - pandas
    - librosa (for audio processing)
    - numpy
    - scikit-learn (optional, for more advanced classification)

### Project Structure
- **main.py**: Core script that orchestrates the processing pipeline
- **audio_processor.py**: Contains the AudioProcessor class for managing feature extraction
- **extractors/**: Directory containing individual feature extractor modules
    - Various feature extractor classes (RMSFeature, MFCCFeature, etc.)
- **hudba/**: Directory for input audio files
- **Output files**:
    - `extrahovane_features.csv`: Raw extracted features
    - `predikovane_typy.csv`: Classification results

### Usage
1. Place audio files in the 'hudba' directory
2. Run the main script:
   ```bash
   python main.py
