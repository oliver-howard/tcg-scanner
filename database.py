import requests
import urllib.parse
import os
import sqlite3
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import imagehash
import numpy as np
from io import BytesIO
import concurrent.futures

BASE_URL = "https://api.tcgdex.net/v2/en"
DB_PATH = "cards.db"
IMAGE_DIR = "card_images"

# --- Setup ResNet-18 for Embeddings ---
# We strip the final classification head to keep the 512-dim embedding
print("Loading ResNet-18 model...")
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Standard ImageNet normalization for ResNet
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cards (
            id TEXT PRIMARY KEY,
            localId TEXT,
            name TEXT,
            category TEXT,
            illustrator TEXT,
            rarity TEXT,
            set_name TEXT,
            variants_normal BOOLEAN,
            variants_reverse BOOLEAN,
            variants_holo BOOLEAN,
            variants_firstEdition BOOLEAN,
            hp INTEGER,
            types TEXT,
            evolveFrom TEXT,
            description TEXT,
            level TEXT,
            stage TEXT,
            phash TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    return conn

def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        return None

def extract_features(img):
    """
    Returns (phash_str, embedding_numpy_array)
    """
    # 1. Perceptual Hash
    phash = str(imagehash.phash(img))
    
    # 2. ResNet Embedding
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    with torch.no_grad():
        output = resnet(input_batch)
    
    # Flatten the (1, 512, 1, 1) tensor to a 1D vector of shape (512,)
    embedding = output.squeeze().numpy()
    
    return phash, embedding

def process_card_worker(card_dict, set_name_text, index, total):
    """
    Worker function to run in a separate thread.
    Downloads the card JSON, downloads the image, and extracts features.
    """
    card_id = card_dict['id']
    name = card_dict.get('name', 'Unknown')
    img_url = card_dict.get('image')

    if not img_url:
        print(f"[{index}/{total}] Skipping {name} (No image URL)")
        return None
        
    full_img_url = f"{img_url}/high.webp"
    
    try:
        card_resp = requests.get(f"{BASE_URL}/cards/{card_id}", timeout=10)
        card_resp.raise_for_status()
        card_detail = card_resp.json()
    except Exception as e:
        print(f"[{index}/{total}] Failed to fetch details for {name}: {e}")
        return None

    print(f"[{index}/{total}] Specs loaded, downloading HD Image & Hashing {name}...")
    img = download_image(full_img_url)
    if not img:
        print(f"[{index}/{total}] Image download failed for {name}.")
        return None
        
    phash, embedding = extract_features(img)
    
    localId = str(card_detail.get('localId', ''))
    category = card_detail.get('category', 'Unknown')
    illustrator = card_detail.get('illustrator', '')
    rarity = card_detail.get('rarity', '')
    
    variants = card_detail.get('variants', {})
    v_normal = variants.get('normal', False)
    v_reverse = variants.get('reverse', False)
    v_holo = variants.get('holo', False)
    v_1st = variants.get('firstEdition', False)
    
    hp = card_detail.get('hp', None)
    types = ",".join(card_detail.get('types', []))
    evolveFrom = card_detail.get('evolveFrom', '')
    description = card_detail.get('description', '')
    level = str(card_detail.get('level', ''))
    stage = card_detail.get('stage', '')
    
    return (
        card_id, localId, name, category, illustrator, rarity, set_name_text,
        v_normal, v_reverse, v_holo, v_1st,
        hp, types, evolveFrom, description, level, stage,
        phash, embedding.tobytes()
    )

def index_set(set_id):
    """
    Fetches all cards in a given set, downloads their images, extracts features, and stores in SQLite.
    """
    print(f"Fetching cards for set '{set_id}'...")
    conn = setup_db()
    cursor = conn.cursor()
    
    url = f"{BASE_URL}/sets/{set_id}"
    response = requests.get(url)
    response.raise_for_status()
    set_data = response.json()
    
    cards = set_data.get('cards', [])
    set_name_text = set_data.get('name', 'Unknown')
    total_cards = len(cards)
    print(f"Found {total_cards} cards in set. Checking existing DB...")
    
    # Pre-filter cards that are already indexed so we don't start useless threads
    cards_to_process = []
    for c in cards:
        cursor.execute("SELECT id FROM cards WHERE id=?", (c['id'],))
        if not cursor.fetchone():
            cards_to_process.append(c)
            
    print(f"{total_cards - len(cards_to_process)} already indexed. {len(cards_to_process)} remaining to process.")
    
    if not cards_to_process:
        print("Indexing complete.")
        return

    # Process remaining cards concurrently (10 threads to respect API limits but be fast)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all jobs
        future_to_card = {
            executor.submit(process_card_worker, c, set_name_text, i+1, len(cards_to_process)): c 
            for i, c in enumerate(cards_to_process)
        }
        
        # As threads finish, gather the results and write to SQLite
        for future in concurrent.futures.as_completed(future_to_card):
            result = future.result()
            if result:
                # result is the tuple of properties
                cursor.execute('''
                    INSERT INTO cards (
                        id, localId, name, category, illustrator, rarity, set_name,
                        variants_normal, variants_reverse, variants_holo, variants_firstEdition,
                        hp, types, evolveFrom, description, level, stage,
                        phash, embedding
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', result)
                conn.commit()

    conn.close()
    print("Indexing complete.")

if __name__ == "__main__":
    # Base Set ID on TCGDex is "base1"
    index_set("bw11")
