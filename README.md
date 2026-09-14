# TCG Scanner

A webcam-based trading card scanner that identifies cards from a live camera feed.

The scanner detects and straightens a card image, then tries a few matching methods in order: perceptual hashing for exact artwork, image embeddings for close matches, and OCR plus an API lookup as a fallback.

## Run

```bash
pip install -r requirements.txt
python main.py
```

A webcam is required. Press `s` to scan the card in the frame and `q` to quit.
