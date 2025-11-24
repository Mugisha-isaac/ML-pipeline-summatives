"""
Audio Talent Classification Pipeline
Processes audio files directly from GitHub, extracts features, and trains a classification model
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import requests
from io import BytesIO
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class GitHubAudioLoader:
    """Loads audio files directly from GitHub repository"""
    
    def __init__(self, repo_url="https://raw.githubusercontent.com/Mugisha-isaac/ML-pipeline-summatives"):
        self.repo_url = repo_url
        self.base_url = f"{repo_url}/main"
    
    def get_file_list_from_github(self, directory="data/test"):
        """
        Get list of audio files from GitHub directory using API
        """
        # Convert raw URL to API URL
        api_url = f"https://api.github.com/repos/Mugisha-isaac/ML-pipeline-summatives/contents/{directory}"
        
        try:
            print(f"Fetching file list from: {api_url}")
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            files_data = response.json()
            # Filter only .wav files
            wav_files = [file['name'] for file in files_data if file['name'].endswith('.wav')]
            print(f"Found {len(wav_files)} .wav files in {directory}")
            return wav_files
        except Exception as e:
            print(f"Error fetching file list from GitHub API: {e}")
            print("Please make sure the repository is public and the path is correct.")
            return []
    
    def load_audio_from_url(self, file_path):
        """Load audio file directly from GitHub URL"""
        url = f"{self.base_url}/{file_path}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Load audio from bytes
            audio_bytes = BytesIO(response.content)
            audio, sr = librosa.load(audio_bytes, sr=22050)
            return audio, sr
        except Exception as e:
            print(f"Error loading {url}: {e}")
            return None, None


class AudioPreprocessor:
    """Handles audio preprocessing and silence removal"""
    
    def __init__(self, sr=22050, top_db=20):
        self.sr = sr
        self.top_db = top_db
    
    def remove_silence(self, audio):
        """Remove silence from audio"""
        non_silent_intervals = librosa.effects.split(audio, top_db=self.top_db)
        if len(non_silent_intervals) == 0:
            return audio
        audio_trimmed = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
        return audio_trimmed
    
    def visualize_waveform(self, audio, sr, title="Waveform"):
        """Display waveform"""
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(audio, sr=sr)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_spectrogram(self, audio, sr, title="Spectrogram"):
        """Display spectrogram"""
        plt.figure(figsize=(12, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()


class AudioAugmenter:
    """Applies audio augmentations"""
    
    def __init__(self, sr=22050):
        self.sr = sr
    
    def pitch_shift(self, audio, n_steps=2):
        """Shift pitch by n_steps semitones"""
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
    
    def time_stretch(self, audio, rate=1.1):
        """Stretch or compress time"""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def add_noise(self, audio, noise_factor=0.005):
        """Add random white noise"""
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
    
    def augment_audio(self, audio, augmentations=['pitch_shift', 'time_stretch']):
        """Apply multiple augmentations"""
        augmented = []
        
        if 'pitch_shift' in augmentations:
            augmented.append(self.pitch_shift(audio, n_steps=np.random.randint(-2, 3)))
        
        if 'time_stretch' in augmentations:
            augmented.append(self.time_stretch(audio, rate=np.random.uniform(0.9, 1.1)))
        
        if 'add_noise' in augmentations:
            augmented.append(self.add_noise(audio))
        
        return augmented


class FeatureExtractor:
    """Extracts audio features using librosa"""
    
    def __init__(self, sr=22050, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def extract_features(self, audio):
        """Extract comprehensive audio features"""
        features = {}
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Energy/RMS
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sr)
        features['tempo'] = tempo
        
        return features
    
    def flatten_features(self, features_dict):
        """Flatten nested feature dictionary"""
        flat_features = {}
        for key, value in features_dict.items():
            if isinstance(value, np.ndarray):
                for i, v in enumerate(value):
                    flat_features[f"{key}_{i}"] = v
            else:
                flat_features[key] = value
        return flat_features


class TalentClassificationPipeline:
    """Main pipeline for processing and feature extraction from GitHub"""
    
    def __init__(self, repo_url="https://raw.githubusercontent.com/Mugisha-isaac/ML-pipeline-summatives",
                 output_csv='audio_features.csv'):
        self.github_loader = GitHubAudioLoader(repo_url)
        self.output_csv = output_csv
        self.preprocessor = AudioPreprocessor()
        self.augmenter = AudioAugmenter()
        self.feature_extractor = FeatureExtractor()
        self.features_list = []
    
    def get_label_from_filename(self, filename):
        """
        Extract label from filename
        Based on the pattern: Ani and Kenshin are good singers, others are bad
        """
        filename_lower = filename.lower()
        # Good singers
        if 'ani' in filename_lower or 'kenshin' in filename_lower:
            return 'good'
        # Bad singers
        else:
            return 'bad'
    
    def process_files_from_github(self, directory="data/test", visualize_samples=3):
        """Process all audio files from GitHub repository"""
        file_list = self.github_loader.get_file_list_from_github(directory)
        
        if not file_list:
            print(f"No files found in {directory}. Skipping.")
            return
        
        print(f"\nProcessing {len(file_list)} files from GitHub: {directory}")
        
        for idx, filename in enumerate(file_list):
            file_path = f"{directory}/{filename}"
            print(f"\n[{idx+1}/{len(file_list)}] Processing: {filename}")
            
            # Load audio from GitHub
            audio, sr = self.github_loader.load_audio_from_url(file_path)
            if audio is None:
                print(f"Skipping {filename} due to loading error")
                continue
            
            # Remove silence
            audio_clean = self.preprocessor.remove_silence(audio)
            print(f"Audio loaded: {len(audio_clean)/sr:.2f}s (after silence removal)")
            
            # Visualize first few samples
            if idx < visualize_samples:
                try:
                    fig1 = self.preprocessor.visualize_waveform(audio_clean, sr, f"Waveform: {filename}")
                    plt.savefig(f"visualizations/waveform_{filename.replace('.wav', '.png')}")
                    plt.close()
                    
                    fig2 = self.preprocessor.visualize_spectrogram(audio_clean, sr, f"Spectrogram: {filename}")
                    plt.savefig(f"visualizations/spectrogram_{filename.replace('.wav', '.png')}")
                    plt.close()
                    print(f"Visualizations saved")
                except Exception as e:
                    print(f"Visualization error: {e}")
            
            # Get label
            file_label = self.get_label_from_filename(filename)
            print(f"Label: {file_label}")
            
            # Extract features from original
            features = self.feature_extractor.extract_features(audio_clean)
            features_flat = self.feature_extractor.flatten_features(features)
            features_flat['filename'] = filename
            features_flat['label'] = file_label
            features_flat['augmentation'] = 'original'
            self.features_list.append(features_flat)
            
            # Apply augmentations
            print(f"Applying augmentations...")
            augmented_audios = self.augmenter.augment_audio(
                audio_clean, 
                augmentations=['pitch_shift', 'time_stretch', 'add_noise']
            )
            
            for aug_idx, aug_audio in enumerate(augmented_audios):
                aug_features = self.feature_extractor.extract_features(aug_audio)
                aug_features_flat = self.feature_extractor.flatten_features(aug_features)
                aug_features_flat['filename'] = f"{filename}_aug{aug_idx}"
                aug_features_flat['label'] = file_label
                aug_features_flat['augmentation'] = f'aug_{aug_idx}'
                self.features_list.append(aug_features_flat)
            
            print(f"Completed: 1 original + {len(augmented_audios)} augmentations")
    
    def save_features(self):
        """Save extracted features to CSV"""
        df = pd.DataFrame(self.features_list)
        df.to_csv(self.output_csv, index=False)
        print(f"\n{'='*60}")
        print(f"Features saved to {self.output_csv}")
        print(f"Total samples (including augmentations): {len(df)}")
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        print(f"{'='*60}")
        return df


class TalentClassifier:
    """Deep learning model for talent classification"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
    
    def build_model(self):
        """Build deep neural network"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.input_dim,)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Remove non-feature columns
        feature_cols = [col for col in df.columns if col not in ['filename', 'label', 'augmentation']]
        X = df[feature_cols].values
        y = df['label'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """Train the model"""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        print("="*60)
        
        return y_pred, y_pred_prob
    
    def plot_training_history(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("Training history plot saved to 'training_history.png'")
        plt.show()
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved to 'confusion_matrix.png'")
        plt.show()
    
    def save_model(self, model_path='models/talent_classifier_model.h5',
                   scaler_path='models/scaler.pkl',
                   label_encoder_path='models/label_encoder.pkl'):
        """Save trained model"""
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        self.model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save scaler and label encoder
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Scaler saved to {scaler_path}")
        print(f"Label encoder saved to {label_encoder_path}")


# Main execution
if __name__ == "__main__":
    # Create output directories
    os.makedirs('visualizations', exist_ok=True)
    
    print("="*60)
    print("AUDIO TALENT CLASSIFICATION PIPELINE")
    print("Loading data from GitHub repository")
    print("="*60)
    
    # Step 1: Process audio files and extract features
    print("\n[STEP 1] Processing audio files and extracting features...")
    pipeline = TalentClassificationPipeline(
        repo_url="https://raw.githubusercontent.com/Mugisha-isaac/ML-pipeline-summatives"
    )
    
    # Process both training and test data from GitHub
    print("\n>>> Processing TRAINING data from GitHub...")
    pipeline.process_files_from_github('data/train', visualize_samples=2)
    
    print("\n>>> Processing TEST data from GitHub...")
    pipeline.process_files_from_github('data/test', visualize_samples=2)
    
    # Save features
    df_features = pipeline.save_features()
    
    if len(df_features) == 0:
        print("No features extracted. Please check the GitHub repository and file paths.")
        exit(1)
    
    # Step 2: Train the model
    print("\n" + "="*60)
    print("[STEP 2] Training classification model...")
    print("="*60)
    classifier = TalentClassifier(input_dim=df_features.shape[1] - 3)  # Exclude metadata columns
    classifier.build_model()
    
    print("\nModel Architecture:")
    classifier.model.summary()
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(df_features)
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train
    print("\nTraining model...")
    classifier.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=32)
    
    # Step 3: Evaluate and visualize
    print("\n" + "="*60)
    print("[STEP 3] Evaluating model...")
    print("="*60)
    y_pred, y_pred_prob = classifier.evaluate(X_test, y_test)
    
    # Plot results
    print("\nGenerating visualizations...")
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(y_test, y_pred)
    
    # Step 4: Save model
    print("\n" + "="*60)
    print("[STEP 4] Saving model...")
    print("="*60)
    classifier.save_model('talent_classifier_model.h5')
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nOutput files created:")
    print("[DONE] audio_features.csv - Extracted features dataset")
    print("[DONE] talent_classifier_model.h5 - Trained model")
    print("[DONE] scaler.pkl - Feature scaler")
    print("[DONE] label_encoder.pkl - Label encoder")
    print("[DONE] training_history.png - Training metrics visualization")
    print("[DONE] confusion_matrix.png - Model performance matrix")
    print("[DONE] visualizations/ - Sample waveforms and spectrograms")
    print("\nYou can now use 'predict_talent.py' to make predictions on new audio files!")
    print("="*60)