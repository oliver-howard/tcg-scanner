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
            name TEXT,
            set_name TEXT,
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
        print(f"Failed to download image {url}: {e}")
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

def index_set(set_id):
    """
    Fetches all cards in a given set, downloads their images, extracts features, and stores in SQLite.
    """
    print(f"Fetching cards for set '{set_id}'...")
    conn = setup_db()
    cursor = conn.cursor()
    
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        
    url = f"{BASE_URL}/sets/{set_id}"
    response = requests.get(url)
    response.raise_for_status()
    set_data = response.json()
    
    cards = set_data.get('cards', [])
    print(f"Found {len(cards)} cards in set. Processing...")
    
    for i, c in enumerate(cards):
        card_id = c['id']
        name = c.get('name', 'Unknown')
        img_url = c.get('image')
        
        # TCGDex image URLs return high-quality if we append /high.webp
        if not img_url:
            print(f"[{i+1}/{len(cards)}] Skipping {name} (No image URL)")
            continue
            
        full_img_url = f"{img_url}/high.webp"
        
        # Check if already indexed
        cursor.execute("SELECT id FROM cards WHERE id=?", (card_id,))
        if cursor.fetchone():
            print(f"[{i+1}/{len(cards)}] Skipping {name} (Already indexed)")
            continue
            
        print(f"[{i+1}/{len(cards)}] Downloading and hashing {name}...")
        img = download_image(full_img_url)
        if img:
            phash, embedding = extract_features(img)
            
            cursor.execute('''
                INSERT INTO cards (id, name, set_name, phash, embedding)
                VALUES (?, ?, ?, ?, ?)
            ''', (card_id, name, set_data.get('name', 'Unknown'), phash, embedding.tobytes()))
            conn.commit()

    conn.close()
    print("Indexing complete.")

if __name__ == "__main__":
    # Base Set ID on TCGDex is "base1"
    index_set("sm9")
