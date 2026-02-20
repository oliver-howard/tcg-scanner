import cv2
import threading
from scanner import find_card_contour, four_point_transform
from hashing import find_by_phash
from embeddings import find_by_embedding
from ocr import extract_card_features
from api import search_card_by_name, get_card_details

current_card_info = None
is_scanning = False

def background_scan(warped_img):
    global current_card_info, is_scanning
    
    try:
        current_card_info = {"status": "[Stage 1] Extracting Perceptual Hash..."}
        
        # 1. Try blazing fast pHash matching for identical art
        hash_match = find_by_phash(warped_img)
        if hash_match:
            current_card_info = {
                "status": "Success (pHash Match)",
                "id": hash_match["id"],
                "name": hash_match["name"],
                "set": hash_match["set_name"],
            }
            return
            
        # 2. Try ResNet Embedding check for similar art / holofoil variations
        current_card_info = {"status": "[Stage 2] Generating Neural Embedding..."}
        embed_match = find_by_embedding(warped_img)
        if embed_match:
            current_card_info = {
                "status": f"Success (Embedding {embed_match['embedding_score']:.2f})",
                "id": embed_match["id"],
                "name": embed_match["name"],
                "set": embed_match["set_name"],
            }
            return
            
        # 3. Fallback to Targeted OCR and API Search
        current_card_info = {"status": "[Stage 3] Targeted OCR reading name..."}
        features = extract_card_features(warped_img)
        
        name = features["name"]
        set_number = features["set_number"]
        
        print("\n--- OCR Region Extraction Results ---")
        print(f"Extracted Name: '{name}'")
        print(f"Extracted Set Number: '{set_number}'")
        print("-------------------------------------\n")
        
        if not name:
            current_card_info = {"status": "Failed: Could not recognize card name."}
            return
            
        current_card_info = {"status": f"Found: {name}, fetching API..."}
        
        cards = search_card_by_name(name)
        if not cards:
            current_card_info = {"status": f"No API results for '{name}'"}
            return
            
        first_card = cards[0]
        details = get_card_details(first_card['id'])
        
        if details:
            current_card_info = {
                "status": "Success (OCR + API)",
                "name": details.get('name', name),
                "set": details.get('set', {}).get('name', 'Unknown'),
                "artist": details.get('illustrator', 'Unknown'),
                "ocr_set_number": features["set_number"]
            }
        else:
            current_card_info = {"status": f"Failed to get detail node for {name}"}
            
    except Exception as e:
        current_card_info = {"status": f"Error: {e}"}
    finally:
        is_scanning = False

def start_scan_thread(warped_img):
    global is_scanning, current_card_info
    if not is_scanning:
        is_scanning = True
        t = threading.Thread(target=background_scan, args=(warped_img.copy(),))
        t.daemon = True
        t.start()

def draw_info_overlay(frame, info):
    if not info:
        return
        
    y_offset = 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (600, 250), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    for key, value in info.items():
        text = f"{key.capitalize()}: {value}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

def main():
    cap = None
    for i in range(5):
        print(f"Trying to open camera at index {i}...")
        temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            if ret:
                cap = temp_cap
                break
            else:
                temp_cap.release()
                
    if cap is None or not cap.isOpened():
        print("Error: Could not open any webcam.")
        return
        
    # Request a 16:9 widescreen resolution to prevent the camera from defaulting to a squashed 4:3 aspect ratio
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    print("=====================================================")
    print("Advanced TCG Scanner Pipeline Active!")
    print("Hold a card up to the camera and wait for the green box.")
    print("Press 's' to scan the card inside the box.")
    print("Press 'q' to quit the application.")
    print("=====================================================")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nError: Failed to grab physical frame from camera.")
            break
            
        contour = find_card_contour(frame)
        flat_card = None
        
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            flat_card = four_point_transform(frame, contour)
            
            # --- Draw OCR Debugging Boxes ---
            fh, fw = flat_card.shape[:2]
            
            # Name bounding box (Red)
            # Left 15% to Right 75%, Top 2% to Top 12%
            cv2.rectangle(flat_card, 
                          (int(fw * 0.15), int(fh * 0.02)), 
                          (int(fw * 0.75), int(fh * 0.12)), 
                          (0, 0, 255), 2)
                          
            # Set Number bounding box (Red)
            # Left 3% to Right 97%, Bottom 88% to Bottom 99%
            cv2.rectangle(flat_card, 
                          (int(fw * 0.03), int(fh * 0.88)), 
                          (int(fw * 0.97), int(fh * 0.99)), 
                          (0, 0, 255), 2)
            
            # Show the cropped perspective card in its own tiny window
            cv2.imshow("Flattened Crop", cv2.resize(flat_card, (240, 340)))
            
        draw_info_overlay(frame, current_card_info)
        cv2.imshow("TCG Scanner", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and flat_card is not None:
            start_scan_thread(flat_card)
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
