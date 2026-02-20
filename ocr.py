import easyocr
import numpy as np
import cv2
import re

# Initialize the EasyOCR reader once to avoid reloading the model on every frame.
print("Initializing EasyOCR... (This may take a moment down downloading models)")
reader = easyocr.Reader(['en'], gpu=False)

def extract_regional_text(image, region_type):
    """
    Extracts text from a highly specific slice of the 480x680 normalized card image.
    region_type options: "name", "set_number"
    """
    height, width = image.shape[:2]
    
    if region_type == "name":
        # Pokemon names are typically at the top 12% of the card
        # Left 15% to Right 75% bounds it away from HP and avoids the Stage icon on the far left
        y1, y2 = int(height * 0.02), int(height * 0.12)
        x1, x2 = int(width * 0.15), int(width * 0.75)
        
    elif region_type == "set_number":
        # Set numbers (e.g., "12/102") can be at the bottom left OR bottom right depending on the era
        # Bottom 12% of the card, full width
        y1, y2 = int(height * 0.88), int(height * 0.99)
        x1, x2 = int(width * 0.03), int(width * 0.97)
    else:
        return None
        
    # Crop the image to just the targeted ribbon
    roi = image[y1:y2, x1:x2]
    
    # Upscale the region by 2x to make incredibly tiny copyright fonts / set numbers legible to EasyOCR
    if region_type == "set_number":
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
    # Run EasyOCR only on the sub-image
    results = reader.readtext(roi, detail=0) # detail=0 returns just the string list
    
    if not results:
        return None
        
    # Join in case it read it as multiple disjoint bounding boxes
    text = " ".join(results).strip()
    return text if len(text) > 0 else None

def extract_card_features(image):
    """
    Returns a dictionary of specifically targeted strings.
    """
    raw_name = extract_regional_text(image, "name")
    name = None
    if raw_name:
        # Filter out common OCR artifacts from evolution stage icons
        name = raw_name
        for noise in ["1GE2", "1GE1", "1GE", "STAGE 2", "STAGE 1", "BASIC", "STAGE "]:
            name = name.replace(noise, "").strip()
            
    raw_set_number = extract_regional_text(image, "set_number")
    
    # Debugging exact string EasyOCR sees before regex
    print(f"\n[DEBUG OCR] Raw text seen at bottom: '{raw_set_number}'")
    
    set_number = None
    if raw_set_number:
        # Use regex to find the precise "number/number" pattern amidst the Copyright/Illustrator noise
        # EasyOCR sometimes injects spaces around the slash, or misreads '/' as 'l', 'I', or '\'
        match = re.search(r'\d+\s*[/\\]\s*\d+', raw_set_number)
        if not match:
            # Fallback for letters mistakenly read as slash
            match = re.search(r'\d+\s*[lI|\\]\s*\d+', raw_set_number)
            
        if match:
            set_number = match.group().replace(" ", "")
            set_number = set_number.replace("l", "/").replace("I", "/").replace("|", "/").replace("\\", "/")
            
    return {
        "name": name,
        "set_number": set_number
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (480, 680)) # Ensure test image is standardized
        if img is not None:
            features = extract_card_features(img)
            print(f"Extracted Features: {features}")
        else:
            print(f"Could not load image at {img_path}")

