"""Training module for model retraining"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

from app.config.settings import TRAINING_EPOCHS, TRAINING_BATCH_SIZE, TRAINING_TEST_SPLIT, SAMPLE_RATE, N_MFCC
from app.utils.audio import FeatureExtractor, AudioPreprocessor

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        self.feature_extractor = FeatureExtractor(sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        self.preprocessor = AudioPreprocessor(sr=SAMPLE_RATE)
        self.last_metrics = {}

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        feature_cols = [col for col in df.columns if col not in ['filename', 'label', 'augmentation']]
        X = df[feature_cols].values
        y = df['label'].values
        y_encoded = self.label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=TRAINING_TEST_SPLIT, random_state=42, stratify=y_encoded
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_model(self, input_dim: int):
        self.model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
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
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              epochs: int = None, batch_size: int = None) -> Dict:
        if epochs is None:
            epochs = TRAINING_EPOCHS
        if batch_size is None:
            batch_size = TRAINING_BATCH_SIZE

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001
        )
        self.history = self.model.fit(
            X_train, y_train, validation_data=(X_test, y_test),
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr], verbose=0
        )
        return self.history.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        self.last_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'class_labels': self.label_encoder.classes_.tolist()
        }
        return self.last_metrics

    def get_feature_count(self, df: pd.DataFrame) -> int:
        feature_cols = [col for col in df.columns if col not in ['filename', 'label', 'augmentation']]
        return len(feature_cols)
