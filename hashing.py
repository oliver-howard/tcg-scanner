import sqlite3
import imagehash
from PIL import Image

DB_PATH = "cards.db"

def find_by_phash(img, max_difference=10):
    """
    Computes the pHash of the camera frame and finds the closest match
    in the local database using Hamming Distance.
    """
    target_phash = imagehash.phash(Image.fromarray(img))
    # Fetch all hashes to compare
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cards")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return None
        
    best_match = None
    min_dist = float('inf')
    
    for row in rows:
        (card_id, localId, name, category, illustrator, rarity, set_name,
         v_normal, v_reverse, v_holo, v_1st,
         hp, types, evolveFrom, description, level, stage,
         stored_phash_str, emb_blob) = row
         
        stored_phash = imagehash.hex_to_hash(stored_phash_str)
        dist = target_phash - stored_phash
        
        if dist < min_dist:
            min_dist = dist
            # Prepare rich dictionary
            best_match = {
                "id": card_id,
                "localId": localId,
                "name": name,
                "category": category,
                "illustrator": illustrator,
                "rarity": rarity,
                "set_name": set_name,
                "variants": {
                    "normal": bool(v_normal),
                    "reverse": bool(v_reverse),
                    "holo": bool(v_holo),
                    "firstEdition": bool(v_1st)
                },
                "hp": hp,
                "types": types.split(",") if types else [],
                "evolveFrom": evolveFrom,
                "description": description,
                "level": level,
                "stage": stage,
                "phash_diff": dist
            }
            
    if min_dist <= max_difference:
        return best_match
    return None
