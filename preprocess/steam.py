import requests
from constants import DAWNTRAIL_ID

def get_steam_reviews(param):
    url = f"https://store.steampowered.com/appreviews/{DAWNTRAIL_ID}"
    response = requests.get(url, param, timeout=10.0)
    return response.json()