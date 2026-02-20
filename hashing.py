import sqlite3
import imagehash
from PIL import Image

DB_PATH = "cards.db"

def find_by_phash(img, max_difference=10):
    """
    Computes the pHash of the camera frame and finds the closest match
    in the local database using Hamming Distance.
    """
    target_hash = imagehash.phash(Image.fromarray(img))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, set_name, phash FROM cards")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return None
        
    best_match = None
    best_diff = float('inf')
    
    for row in rows:
        card_id, name, set_name, db_phash_str = row
        db_hash = imagehash.hex_to_hash(db_phash_str)
        
        diff = target_hash - db_hash
        if diff < best_diff:
            best_diff = diff
            best_match = {
                "id": card_id,
                "name": name,
                "set_name": set_name,
                "phash_diff": diff
            }
            
    if best_diff <= max_difference:
        return best_match
    return None
