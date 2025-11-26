"""Audio processing utilities"""
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Tuple, Dict, List

matplotlib.use('Agg')

class AudioPreprocessor:
    def __init__(self, sr: int = 22050, top_db: int = 20):
        self.sr = sr
        self.top_db = top_db

    def remove_silence(self, audio: np.ndarray) -> np.ndarray:
        non_silent_intervals = librosa.effects.split(audio, top_db=self.top_db)
        if len(non_silent_intervals) == 0:
            return audio
        return np.concatenate([audio[start:end] for start, end in non_silent_intervals])

    def load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        try:
            print(f"[AUDIO] Loading audio from: {file_path}")
            audio, sr = librosa.load(file_path, sr=self.sr)
            print(f"[AUDIO] Audio loaded successfully: shape={audio.shape}, sr={sr}")
            return audio, sr
        except Exception as e:
            error_msg = str(e)
            print(f"[AUDIO] Error loading audio: {error_msg}")
            # Re-raise with more context
            raise ValueError(f"Error loading audio file: {error_msg}")

class AudioAugmenter:
    def __init__(self, sr: int = 22050):
        self.sr = sr

    def pitch_shift(self, audio: np.ndarray, n_steps: int = 2) -> np.ndarray:
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)

    def time_stretch(self, audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
        return librosa.effects.time_stretch(audio, rate=rate)

    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise

    def augment_audio(self, audio: np.ndarray, augmentations: List[str] = None) -> List[np.ndarray]:
        if augmentations is None:
            augmentations = ['pitch_shift', 'time_stretch']
        augmented = []
        if 'pitch_shift' in augmentations:
            augmented.append(self.pitch_shift(audio, n_steps=np.random.randint(-2, 3)))
        if 'time_stretch' in augmentations:
            augmented.append(self.time_stretch(audio, rate=np.random.uniform(0.9, 1.1)))
        if 'add_noise' in augmentations:
            augmented.append(self.add_noise(audio))
        return augmented

class FeatureExtractor:
    def __init__(self, sr: int = 22050, n_mfcc: int = 13):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def extract_features(self, audio: np.ndarray) -> Dict:
        features = {}
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # Use faster tempo estimation with timeout to prevent hanging on Render
        try:
            # Use onset_strength for faster tempo estimation instead of beat_track
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
            if len(onset_env) > 0:
                # Estimate tempo from energy flux without expensive beat tracking
                tempogram = librosa.feature.tempogram(onset_env=onset_env, sr=self.sr)
                tempo = librosa.feature.tempo(onset_env=onset_env, sr=self.sr)[0]
            else:
                tempo = 0.0
        except Exception as e:
            print(f"[FEATURE_EXTRACT] Warning: Could not extract tempo: {e}. Using default value.")
            tempo = 0.0
        
        features['tempo'] = tempo
        return features

    def flatten_features(self, features_dict: Dict) -> Dict:
        flat_features = {}
        for key, value in features_dict.items():
            if isinstance(value, np.ndarray):
                for i, v in enumerate(value):
                    flat_features[f"{key}_{i}"] = float(v)
            else:
                # For scalar values like tempo
                flat_features[f"{key}_0"] = float(value)
        return flat_features
