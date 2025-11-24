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
            audio_clean = self.preprocessor.remove_silence(audio)
            features = self.feature_extractor.extract_features(audio_clean)
            features_flat = self.feature_extractor.flatten_features(features)
            feature_array = np.array([list(features_flat.values())])
            features_scaled = self.scaler.transform(feature_array)
            prediction_prob = self.model.predict(features_scaled, verbose=0)[0][0]
            prediction_class = int(prediction_prob > 0.5)
            predicted_label = self.label_encoder.inverse_transform([prediction_class])[0]
            return {
                'label': predicted_label,
                'confidence': float(prediction_prob if prediction_class == 1 else 1 - prediction_prob),
                'probability_good': float(prediction_prob),
                'probability_bad': float(1 - prediction_prob)
            }
        except Exception as e:
            raise ValueError(f"Error during prediction: {e}")

    def is_model_ready(self) -> bool:
        return self.model_loaded and self.model is not None
