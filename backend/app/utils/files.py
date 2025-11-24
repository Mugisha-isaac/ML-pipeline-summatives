"""File handling utilities"""
import os
import shutil
from pathlib import Path
from typing import List, Tuple
from app.config.settings import UPLOAD_DIR, ALLOWED_AUDIO_FORMATS, MAX_UPLOAD_SIZE

def save_upload_file(file_path: str, destination: str = None) -> str:
    if not destination:
        destination = str(UPLOAD_DIR)
    os.makedirs(destination, exist_ok=True)
    filename = os.path.basename(file_path)
    dest_path = os.path.join(destination, filename)
    shutil.copy2(file_path, dest_path)
    return dest_path

def get_audio_files(directory: str = None) -> List[str]:
    if not directory:
        directory = str(UPLOAD_DIR)
    audio_files = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if Path(file).suffix.lower() in ALLOWED_AUDIO_FORMATS:
                audio_files.append(os.path.join(directory, file))
    return audio_files

def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    if not os.path.exists(file_path):
        return False, "File does not exist"
    ext = Path(file_path).suffix.lower()
    if ext not in ALLOWED_AUDIO_FORMATS:
        return False, f"Unsupported format. Allowed: {', '.join(ALLOWED_AUDIO_FORMATS)}"
    file_size = os.path.getsize(file_path)
    if file_size > MAX_UPLOAD_SIZE:
        return False, f"File too large. Max size: {MAX_UPLOAD_SIZE / (1024*1024):.1f} MB"
    return True, "Valid"
