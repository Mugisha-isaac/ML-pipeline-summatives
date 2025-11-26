"""Model management utilities"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any
from tensorflow import keras

from app.config.settings import MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, SAMPLE_RATE, N_MFCC
from app.utils.audio import AudioPreprocessor, FeatureExtractor

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.model_loaded = False
        self.preprocessor = AudioPreprocessor(sr=SAMPLE_RATE)
        self.feature_extractor = FeatureExtractor(sr=SAMPLE_RATE, n_mfcc=N_MFCC)

    def load_model(self) -> bool:
        try:
            if os.path.exists(str(MODEL_PATH)):
                self.model = keras.models.load_model(str(MODEL_PATH))
            if os.path.exists(str(SCALER_PATH)):
                with open(str(SCALER_PATH), 'rb') as f:
                    self.scaler = pickle.load(f)
            if os.path.exists(str(LABEL_ENCODER_PATH)):
                with open(str(LABEL_ENCODER_PATH), 'rb') as f:
                    self.label_encoder = pickle.load(f)
            if self.model and self.scaler and self.label_encoder:
                self.model_loaded = True
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def save_model(self) -> bool:
        try:
            if self.model:
                os.makedirs(str(MODEL_PATH.parent), exist_ok=True)
                self.model.save(str(MODEL_PATH))
            if self.scaler:
                with open(str(SCALER_PATH), 'wb') as f:
                    pickle.dump(self.scaler, f)
            if self.label_encoder:
                with open(str(LABEL_ENCODER_PATH), 'wb') as f:
                    pickle.dump(self.label_encoder, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def predict(self, file_path: str) -> Dict[str, Any]:
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        try:
            audio, sr = self.preprocessor.load_audio_file(file_path)
            print(f"[PREDICT] Audio loaded: shape={audio.shape}")
            
            audio_clean = self.preprocessor.remove_silence(audio)
            print(f"[PREDICT] After silence removal: shape={audio_clean.shape}")
            
            features = self.feature_extractor.extract_features(audio_clean)
            print(f"[PREDICT] Features extracted: {len(features)} feature keys")
            
            features_flat = self.feature_extractor.flatten_features(features)
            print(f"[PREDICT] Features flattened: {len(features_flat)} total features")
            
            # Create feature array in the EXACT order as CSV columns
            # Expected order: mfcc_mean_0-12, mfcc_std_0-12, spectral_centroid_mean, 
            # spectral_centroid_std, spectral_rolloff_mean, spectral_rolloff_std, 
            # zcr_mean, zcr_std, rms_mean, rms_std, chroma_mean, chroma_std, tempo_0
            feature_order = []
            for i in range(13):
                feature_order.append(f"mfcc_mean_{i}")
            for i in range(13):
                feature_order.append(f"mfcc_std_{i}")
            feature_order.extend([
                "spectral_centroid_mean", "spectral_centroid_std",
                "spectral_rolloff_mean", "spectral_rolloff_std",
                "zcr_mean", "zcr_std",
                "rms_mean", "rms_std",
                "chroma_mean", "chroma_std",
                "tempo_0"
            ])
            
            feature_values = []
            for fname in feature_order:
                if fname in features_flat:
                    feature_values.append(features_flat[fname])
                else:
                    print(f"[PREDICT] WARNING: Missing feature {fname}")
                    feature_values.append(0.0)  # Default to 0 if missing
            
            feature_array = np.array([feature_values])
            print(f"[PREDICT] Feature array shape: {feature_array.shape} (expected: (1, 37))")
            
            if feature_array.shape[1] != 37:
                raise ValueError(f"Feature dimension mismatch: expected 37, got {feature_array.shape[1]}")
            
            features_scaled = self.scaler.transform(feature_array)
            print(f"[PREDICT] Scaled features shape: {features_scaled.shape}")
            
            prediction_prob = self.model.predict(features_scaled, verbose=0)[0][0]
            print(f"[PREDICT] Prediction probability: {prediction_prob}")
            
            prediction_class = int(prediction_prob > 0.5)
            predicted_label = self.label_encoder.inverse_transform([prediction_class])[0]
            print(f"[PREDICT] Predicted label: {predicted_label}")
            
            return {
                'label': predicted_label,
                'confidence': float(prediction_prob if prediction_class == 1 else 1 - prediction_prob),
                'probability_good': float(prediction_prob),
                'probability_bad': float(1 - prediction_prob)
            }
        except Exception as e:
            print(f"[PREDICT] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error during prediction: {e}")

    def is_model_ready(self) -> bool:
        return self.model_loaded and self.model is not None
