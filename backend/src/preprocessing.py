"""
Preprocessing module for audio talent classification
Handles GitHub integration, audio preprocessing, augmentation, and feature extraction
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


class GitHubAudioLoader:
    """Loads audio files directly from GitHub repository"""
    
    def __init__(self, repo_url="https://raw.githubusercontent.com/Mugisha-isaac/ML-pipeline-summatives"):
        self.repo_url = repo_url
        self.base_url = f"{repo_url}/main"
    
    def get_file_list_from_github(self, directory="data/test"):
        """Get list of audio files from GitHub directory using API"""
        api_url = f"https://api.github.com/repos/Mugisha-isaac/ML-pipeline-summatives/contents/{directory}"
        
        try:
            print(f"Fetching file list from: {api_url}")
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            files_data = response.json()
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
        """Extract label from filename"""
        filename_lower = filename.lower()
        if 'ani' in filename_lower or 'kenshin' in filename_lower:
            return 'good'
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
            
            audio, sr = self.github_loader.load_audio_from_url(file_path)
            if audio is None:
                print(f"Skipping {filename} due to loading error")
                continue
            
            audio_clean = self.preprocessor.remove_silence(audio)
            print(f"Audio loaded: {len(audio_clean)/sr:.2f}s (after silence removal)")
            
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
            
            file_label = self.get_label_from_filename(filename)
            print(f"Label: {file_label}")
            
            features = self.feature_extractor.extract_features(audio_clean)
            features_flat = self.feature_extractor.flatten_features(features)
            features_flat['filename'] = filename
            features_flat['label'] = file_label
            features_flat['augmentation'] = 'original'
            self.features_list.append(features_flat)
            
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