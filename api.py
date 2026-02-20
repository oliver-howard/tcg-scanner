import requests
import urllib.parse
from thefuzz import process

BASE_URL = "https://api.tcgdex.net/v2/en"

def search_card_by_name(name):
    """
    Search for a card by exact or partial name.
    Returns a list of card summary objects.
    """
    encoded_name = urllib.parse.quote(name)
    url = f"{BASE_URL}/cards?name={encoded_name}"
    
    # Try an exact partial-name match first
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        results = response.json()
        if results:
            return results
    except requests.exceptions.RequestException as e:
        print(f"Exact search failed: {e}")
        
    # If no results or error, fall back to fetching all cards and fuzzy matching
    print(f"No direct hits for '{name}'. Attempting fuzzy matching...")
    all_cards_url = f"{BASE_URL}/cards"
    try:
        response = requests.get(all_cards_url, timeout=15)
        response.raise_for_status()
        all_cards = response.json()
        
        # Extract just the names for thefuzz
        card_names = [c.get('name', '') for c in all_cards if c.get('name')]
        
        # Get the top match
        # limit=1 returns a tuple (matched_string, score)
        best_match = process.extractOne(name, card_names)
        
        if best_match and best_match[1] > 70: # 70% confidence threshold
            matched_name = best_match[0]
            print(f"Fuzzy matched '{name}' to '{matched_name}' (Score: {best_match[1]})")
            
            # Now search API again with corrected name
            corrected_url = f"{BASE_URL}/cards?name={urllib.parse.quote(matched_name)}"
            final_response = requests.get(corrected_url, timeout=10)
            final_response.raise_for_status()
            return final_response.json()
            
        print(f"Could not find a confident fuzzy match for '{name}'")
        return []
        
    except requests.exceptions.RequestException as e:
        print(f"Fuzzy fallback failed: {e}")
        return []

def get_card_details(card_id):
    """
    Get full details for a specific card by its ID.
    """
    url = f"{BASE_URL}/cards/{card_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching details for card ID '{card_id}': {e}")
        return None

if __name__ == "__main__":
    print("Searching for Charizard...")
    cards = search_card_by_name("Charizard")
    if cards:
        first_card = cards[0]
        print(f"Found {len(cards)} cards. Fetching details for first one: {first_card['id']}")
        details = get_card_details(first_card['id'])
        if details:
            print(f"Name: {details.get('name')}")
            print(f"Set: {details.get('set', {}).get('name')}")
            print(f"Artist: {details.get('illustrator')}")
            print(f"Rarity: {details.get('rarity')}")
    else:
        print("No cards found.")
