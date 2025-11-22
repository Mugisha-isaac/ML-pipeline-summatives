"""
Prediction script for trained talent classifier
Use this to predict singing talent on new audio files
Loads model directly from GitHub repository
"""

import numpy as np
import librosa
import pickle
from tensorflow import keras
import sys
import os
import requests
from io import BytesIO

class TalentPredictor:
    """Predict singing talent from audio files"""
    
    def __init__(self, 
                 repo_url="https://raw.githubusercontent.com/Mugisha-isaac/ML-pipeline-summatives/main",
                 model_path='models/talent_classifier_model.h5', 
                 scaler_path='models/scaler.pkl', 
                 label_encoder_path='models/label_encoder.pkl',
                 load_from_github=True):
        """
        Initialize predictor with trained model and preprocessors
        
        Args:
            repo_url: GitHub repository raw URL
            model_path: Path to model file (local or in repo)
            scaler_path: Path to scaler file (local or in repo)
            label_encoder_path: Path to label encoder file (local or in repo)
            load_from_github: If True, load from GitHub; if False, load from local
        """
        self.repo_url = repo_url
        self.load_from_github = load_from_github
        
        if load_from_github:
            print("Loading model from GitHub repository...")
            self.model = self._load_model_from_github(model_path)
            self.scaler = self._load_pickle_from_github(scaler_path)
            self.label_encoder = self._load_pickle_from_github(label_encoder_path)
            print("Model loaded successfully from GitHub")
        else:
            print("Loading model from local files...")
            # Check if files exist locally
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            if not os.path.exists(label_encoder_path):
                raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
            
            # Load from local files
            self.model = keras.models.load_model(model_path)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print("Model loaded successfully from local files")
        
        self.sr = 22050
        self.n_mfcc = 13
    
    def _load_model_from_github(self, model_path):
        """Load Keras model from GitHub"""
        url = f"{self.repo_url}/{model_path}"
        print(f"Downloading model from: {url}")
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Save temporarily and load
            temp_path = 'temp_model.h5'
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            model = keras.models.load_model(temp_path)
            os.remove(temp_path)
            return model
        except Exception as e:
            raise Exception(f"Error loading model from GitHub: {e}")
    
    def _load_pickle_from_github(self, pickle_path):
        """Load pickle file from GitHub"""
        url = f"{self.repo_url}/{pickle_path}"
        print(f"Downloading from: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Load pickle from bytes
            obj = pickle.loads(response.content)
            return obj
        except Exception as e:
            raise Exception(f"Error loading pickle from GitHub: {e}")
    
    def preprocess_audio(self, file_path):
        """Load and preprocess audio file"""
        # Load audio
        audio, sr = librosa.load(file_path, sr=self.sr)
        
        # Remove silence
        non_silent_intervals = librosa.effects.split(audio, top_db=20)
        if len(non_silent_intervals) == 0:
            audio_clean = audio
        else:
            audio_clean = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
        
        return audio_clean
    
    def extract_features(self, audio):
        """Extract features from audio"""
        features = {}
        
        # MFCCs
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
        """Flatten feature dictionary to array"""
        flat_features = []
        for key in sorted(features_dict.keys()):
            value = features_dict[key]
            if isinstance(value, np.ndarray):
                flat_features.extend(value)
            else:
                flat_features.append(value)
        return np.array(flat_features)
    
    def predict(self, file_path):
        """Predict talent level for audio file"""
        # Preprocess
        audio = self.preprocess_audio(file_path)
        
        # Extract features
        features = self.extract_features(audio)
        features_flat = self.flatten_features(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features_flat.reshape(1, -1))
        
        # Predict
        prediction_prob = self.model.predict(features_scaled, verbose=0)[0][0]
        prediction_class = int(prediction_prob > 0.5)
        predicted_label = self.label_encoder.inverse_transform([prediction_class])[0]
        
        return {
            'label': predicted_label,
            'confidence': float(prediction_prob if prediction_class == 1 else 1 - prediction_prob),
            'probability_good': float(prediction_prob),
            'probability_bad': float(1 - prediction_prob)
        }
    
    def predict_batch(self, file_paths):
        """Predict for multiple files"""
        results = []
        for file_path in file_paths:
            try:
                result = self.predict(file_path)
                result['file'] = file_path
                results.append(result)
                confidence_pct = result['confidence'] * 100
                print(f"[SUCCESS] {file_path}: {result['label']} (confidence: {confidence_pct:.2f}%)")
            except Exception as e:
                print(f"[ERROR] {file_path}: {e}")
        return results


# Example usage
if __name__ == "__main__":
    
    # Initialize predictor
    print("="*60)
    print("Audio Talent Predictor")
    print("="*60)
    print()
    
    # Check command line arguments
    load_from_github = True
    if '--local' in sys.argv:
        load_from_github = False
        sys.argv.remove('--local')
    
    try:
        predictor = TalentPredictor(
            repo_url="https://raw.githubusercontent.com/Mugisha-isaac/ML-pipeline-summatives/main",
            model_path='models/talent_classifier_model.h5',
            scaler_path='models/scaler.pkl',
            label_encoder_path='models/label_encoder.pkl',
            load_from_github=load_from_github
        )
        print()
    except Exception as e:
        print(f"[ERROR] {e}")
        print()
        print("Please ensure:")
        if load_from_github:
            print("1. The model files exist in the GitHub repository under 'models/' directory")
            print("2. The repository is public")
            print("3. You have internet connection")
            print()
            print("Or use --local flag to load from local files:")
            print("  python prediction.py --local path/to/audio.wav")
        else:
            print("1. Run 'audio_talent_pipeline.py' first to train and save the model")
            print("2. Model files exist in the current directory")
        sys.exit(1)
    
    # Predict on single file
    if len(sys.argv) > 1:
        # Command line usage: python prediction.py path/to/audio.wav
        file_path = sys.argv[1]
        
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            sys.exit(1)
        
        print(f"Predicting talent for: {file_path}")
        print("-"*60)
        
        try:
            result = predictor.predict(file_path)
            
            print()
            print("="*60)
            print("PREDICTION RESULTS")
            print("="*60)
            print(f"File: {file_path}")
            print(f"Talent Level: {result['label'].upper()}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            print()
            print("Probabilities:")
            print(f"  Good Singer: {result['probability_good']*100:.2f}%")
            print(f"  Bad Singer:  {result['probability_bad']*100:.2f}%")
            print("="*60)
        except Exception as e:
            print()
            print(f"[ERROR] Error during prediction: {e}")
            sys.exit(1)
    else:
        # No file provided - show usage
        print("Usage:")
        print("  python prediction.py <audio_file.wav>")
        print("  python prediction.py --local <audio_file.wav>  (load model from local files)")
        print()
        print("Examples:")
        print("  python prediction.py my_singing.wav")
        print("  python prediction.py data/test/Ani_1.wav")
        print("  python prediction.py --local my_singing.wav")
        print()
        print("For batch prediction, modify the script to use predict_batch():")
        print("  files = ['audio1.wav', 'audio2.wav']")
        print("  results = predictor.predict_batch(files)")
        print("="*60)