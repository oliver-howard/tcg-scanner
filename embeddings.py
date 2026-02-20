import sqlite3
import numpy as np
import faiss
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

DB_PATH = "cards.db"

# Initialize ResNet and FAISS lazily to improve startup time
resnet = None
preprocess = None
faiss_index = None
card_metadata = []

def init_models():
    global resnet, preprocess, faiss_index, card_metadata
    
    if resnet is not None:
        return
        
    print("Loading Local Embedding Models and FAISS Index...")
    
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load all embeddings from SQLite into memory for FAISS
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, set_name, embedding FROM cards")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("Warning: Database is empty. Please run database.py first.")
        return
        
    dimension = 512 # ResNet-18 pooled dim
    faiss_index = faiss.IndexFlatIP(dimension) # Inner Product/Cosine similarity
    faiss_index = faiss.IndexIDMap(faiss_index)
    
    embeddings = []
    ids = []
    
    for i, row in enumerate(rows):
        card_id, name, set_name, emb_blob = row
        # Reconstruct numpy array from BLOB
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        
        # Normalize for Cosine Similarity (instead of L2 distance)
        faiss.normalize_L2(emb.reshape(1, -1))
        
        embeddings.append(emb)
        ids.append(i)
        card_metadata.append({
            "id": card_id,
            "name": name,
            "set_name": set_name
        })
        
    faiss_index.add_with_ids(np.array(embeddings), np.array(ids))

def find_by_embedding(img_array):
    """
    Computes ResNet embedding of frame and queries FAISS index.
    """
    init_models()
    if faiss_index is None:
        return None
        
    img = Image.fromarray(img_array).convert('RGB')
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = resnet(input_batch)
        
    query_emb = output.squeeze().numpy().astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_emb)
    
    # Search for top 1 match
    distances, indices = faiss_index.search(query_emb, 1)
    
    best_idx = indices[0][0]
    score = distances[0][0] # 1.0 is identical
    
    if best_idx != -1 and score > 0.85: # Threshold for similarity
        match = card_metadata[best_idx].copy()
        match['embedding_score'] = float(score)
        return match
        
    return None
