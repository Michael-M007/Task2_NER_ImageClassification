import argparse
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForTokenClassification
import inference_ner  # Import your NER inference script
import inference_classifier  # Import your image classification inference script

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def run_pipeline(text, image_path):
    """
    # Extract animal names from text using NER model
    detected_animals_text = inference_ner.predict_animals(text)
    
    if not detected_animals_text:
        print("No animals detected in the text.")
        return False
    
    # Load and classify image using CNN model
    image = load_image(image_path)
    detected_animal_image = inference_classifier.predict_animal(image)
    
    if not detected_animal_image:
        print("No animals detected in the image.")
        return False
    
    # Compare results
    print(f"NER detected: {detected_animals_text}, Image classified as: {detected_animal_image}")
    return detected_animal_image in detected_animals_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animal Recognition Pipeline")
    parser.add_argument("text", type=str, help="Input text describing the image")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()
    
    result = run_pipeline(args.text, args.image_path)
    print(f"Match result: {result}")

