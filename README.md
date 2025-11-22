# Audio Talent Classification Pipeline

A machine learning pipeline for classifying audio talent based on voice features extracted from audio files. This project processes audio files directly from GitHub, extracts comprehensive audio features, and trains a deep learning classification model.

## Overview

This pipeline automates the process of:
- Loading audio files from GitHub repository
- Preprocessing audio data (silence removal, normalization)
- Augmenting audio data for better model generalization
- Extracting audio features (MFCC, spectral features, etc.)
- Training a neural network classifier
- Evaluating model performance

## Features

### Audio Processing
- Automatic silence removal from audio files
- Waveform and spectrogram visualization
- Audio augmentation (pitch shifting, time stretching, noise addition)

### Feature Extraction
- Mel-Frequency Cepstral Coefficients (MFCC)
- Spectral centroids and rolloff
- Zero-crossing rate
- Energy/RMS features
- Chroma features
- Tempo detection

### Model Architecture
- Deep neural network with 5 hidden layers
- Batch normalization and dropout for regularization
- Early stopping to prevent overfitting
- Learning rate reduction on validation loss plateau

### Evaluation Metrics
- Accuracy, Precision, and Recall
- Confusion matrix visualization
- Training history plots

## Project Structure

```
.
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── src/
│   ├── model.py                       # Main pipeline and model classes
│   ├── preprocessing.py               # Audio preprocessing utilities
│   └── prediction.py                  # Prediction on new audio files
├── notebook/
│   └── Audio_talent_classification_pipeline.ipynb  # Jupyter notebook
├── data/
│   ├── train/                         # Training audio files
│   ├── test/                          # Test audio files
│   └── visualisation/                 # Visualization outputs
└── models/
    ├── audio_features.csv             # Extracted features
    ├── talent_classifier_model.h5     # Trained model
    ├── scaler.pkl                     # Feature scaler
    └── label_encoder.pkl              # Label encoder
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Steps

1. Clone the repository:
```bash
git clone https://github.com/Mugisha-isaac/ML-pipeline-summatives.git
cd "talent discovery"
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

```bash
python src/model.py
```

This will:
1. Load audio files from GitHub repository
2. Extract audio features
3. Train the classification model
4. Generate evaluation metrics and visualizations
5. Save the trained model and feature scaler

### Using the Jupyter Notebook

```bash
jupyter notebook notebook/Audio_talent_classification_pipeline.ipynb
```

### Making Predictions on New Audio

```python
from src.prediction import predict_audio

# Predict on a single audio file
result = predict_audio('path/to/audio.wav')
print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2%})")
```

## Dependencies

- **numpy** (>=1.21.0) - Numerical computing
- **pandas** (>=1.3.0) - Data manipulation and analysis
- **librosa** (>=0.9.2) - Audio analysis and feature extraction
- **matplotlib** (>=3.4.0) - Data visualization
- **seaborn** (>=0.11.0) - Statistical visualization
- **scikit-learn** (>=1.0.0) - Machine learning utilities
- **tensorflow** (>=2.10.0) - Deep learning framework
- **soundfile** (>=0.11.0) - Audio I/O

## Model Performance

The trained model provides:
- Binary classification (good/bad talent)
- Stratified train-test split (80/20)
- Feature scaling using StandardScaler
- Label encoding for classification

### Output Files

After running the pipeline, the following files are generated:

- `audio_features.csv` - Extracted features dataset with labels
- `talent_classifier_model.h5` - Trained neural network model
- `scaler.pkl` - Fitted StandardScaler for feature normalization
- `label_encoder.pkl` - Fitted LabelEncoder for class labels
- `training_history.png` - Training metrics visualization
- `confusion_matrix.png` - Model performance matrix
- `visualizations/` - Sample waveforms and spectrograms

## Configuration

Key parameters can be adjusted in `src/model.py`:

- **Sample rate**: Default 22050 Hz
- **MFCC coefficients**: Default 13
- **Silence threshold**: Default 20 dB
- **Train epochs**: Default 100
- **Batch size**: Default 32
- **Test split**: Default 0.2 (20%)

## Data Requirements

Audio files should be:
- In WAV, MP3, or FLAC format
- Properly labeled in directory structure
- Organized as: `data/[train|test]/[label]_*.wav`

## License

This project is available under the MIT License.

## Author

Isaac Mugisha

## Support

For issues or questions, please open an issue on the GitHub repository.

## Acknowledgments

- Built with TensorFlow and Keras
- Audio processing powered by Librosa
- Machine learning utilities from scikit-learn
