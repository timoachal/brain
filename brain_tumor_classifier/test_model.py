#!/usr/bin/env python3
import os
import sys
import django

# Add the project directory to Python path
sys.path.append('/home/ubuntu/brain_tumor_classifier')

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'brain_tumor_classifier.settings')
django.setup()

from classifier.ml_service import classifier

def test_model():
    print("Testing brain tumor classification model...")
    
    # Test model loading
    if classifier.model is None:
        print("❌ Model failed to load")
        return False
    else:
        print("✅ Model loaded successfully")
    
    # Print model summary
    try:
        print("\nModel Architecture:")
        print("Input shape:", classifier.model.input_shape)
        print("Output shape:", classifier.model.output_shape)
        print("Number of parameters:", classifier.model.count_params())
    except Exception as e:
        print(f"Could not get model details: {e}")
    
    print(f"\nClass names: {classifier.class_names}")
    
    return True

if __name__ == "__main__":
    test_model()

