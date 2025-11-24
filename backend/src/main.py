"""
Main entry point for Audio Talent Classification Pipeline
This script orchestrates the entire pipeline from data loading to model training
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import AudioPreprocessor, GitHubAudioLoader, TalentClassificationPipeline
from model import TalentClassifier

def main():
    """Main execution function"""
    # Create output directories
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
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
        return 1
    
    # Step 2: Train the model
    print("\n" + "="*60)
    print("[STEP 2] Training classification model...")
    print("="*60)
    classifier = TalentClassifier(input_dim=df_features.shape[1] - 3)
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
    classifier.save_model(
        model_path='models/talent_classifier_model.h5',
        scaler_path='models/scaler.pkl',
        label_encoder_path='models/label_encoder.pkl'
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nOutput files created:")
    print("- audio_features.csv")
    print("- models/talent_classifier_model.h5")
    print("- models/scaler.pkl")
    print("- models/label_encoder.pkl")
    print("- training_history.png")
    print("- confusion_matrix.png")
    print("- visualizations/")
    print("\nNext steps:")
    print("1. Review the training metrics and confusion matrix")
    print("2. Use prediction.py to make predictions on new audio files")
    print("3. Upload the models/ directory to GitHub if you want to use GitHub-based predictions")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())