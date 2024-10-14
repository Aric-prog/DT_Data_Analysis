import urllib.parse
import requests
import datetime
import logging
import os
import json
import urllib
from constants import FFXIV_ID, STEAM_API_KEY

def get_user_playtime(user_id, game_id):
    "Returns FFXIV Online Steam playtime in minutes"

    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
    params = {
        "key": STEAM_API_KEY,
        "input_json": json.dumps(
            {
                "steamid": user_id,
                "appids_filter": [game_id],
            }
        ),
    }       

    try:
        response = requests.get(url, params, timeout=10.0)
    except Exception as e:
        return None, None

    if response.status_code != 200:
        return None, None

    try:
        playtime = response.json()["response"]["games"][0]["playtime_forever"]
        last_2_weeks = response.json()["response"]["games"][0]["playtime_2weeks"]
    except Exception as e:
        return None, None

    return playtime, last_2_weeks