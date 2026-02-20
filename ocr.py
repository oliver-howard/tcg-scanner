import easyocr
import numpy as np
import cv2

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
        # Left 10% to Right 70% bounds it away from HP and type icons
        y1, y2 = int(height * 0.02), int(height * 0.12)
        x1, x2 = int(width * 0.10), int(width * 0.70)
        
    elif region_type == "set_number":
        # Set numbers (e.g., "12/102") are at the very bottom corner, usually right
        # Bottom 7% of the card, right 30%
        y1, y2 = int(height * 0.93), int(height * 0.98)
        x1, x2 = int(width * 0.50), int(width * 0.95)
    else:
        return None
        
    # Crop the image to just the targeted ribbon
    roi = image[y1:y2, x1:x2]
    
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
    return {
        "name": extract_regional_text(image, "name"),
        "set_number": extract_regional_text(image, "set_number")
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

