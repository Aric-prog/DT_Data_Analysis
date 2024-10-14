import datetime
import json
import urllib
import pandas as pd
import os
import csv

from preprocess.user import get_user_playtime
from preprocess.steam import get_steam_reviews
from constants import DAWNTRAIL_ID, FFXIV_ID


# Special thanks to u/lion_rouge for this piece of code
def get_reviews(review_type):
    """Gets reviews from steam, """
    reviews = []

    params = {
        "json": 1,
        "filter": "recent",
        "review_type": review_type,
        "purchase_type": "steam",
        "start_date": datetime.datetime(2024, 7, 2).timestamp(),  # July 2 (release date)
        "date_range_type": "include",
        "language": "all",
        "num_per_page": 100,
        "cursor" : "*"
    }

    cursor = ""
    # Main Loop
    i = 1
    while True:
        print(f"Page {i}")
        
        data = get_steam_reviews(params)

        for j in data["reviews"]:
            playtime, last_2_weeks = get_user_playtime(j["author"]["steamid"], FFXIV_ID)
            if playtime is not None:
                j["author"]["playtime_forever"] = playtime
                j["author"]["playtime_last_two_weeks"] = last_2_weeks

        for review in data["reviews"]:
            reviews.append(review)

        if data["cursor"] == cursor:
            break

        cursor = data["cursor"]
        params["cursor"] = cursor
        i += 1
        
    with open(f"./data/dawntrail_{review_type}_reviews.json", 'w') as f:
        json.dump(reviews, f)
    
    to_csv(f"./data/dawntrail_{review_type}_reviews.json", review_type)

def to_csv(path, review_type):
    output_csv = open(f'dawntrail_reviews.csv', 'a')
    
    writer = csv.writer(output_csv, delimiter=',')
    if(not os.path.exists(path)):
        writer.writerow(
            [
                'author_id',
                'playtime_forever',
                'playtime_two_weeks',
                'language',
                'review',
                'timestamp_created', 
                'votes_up', 
                'votes_funny',
                'review_type'
            ]
        )
    json_file = json.load(open(f"{path}", 'r'))
    for line in json_file:
        review = line["review"].replace("\n", "")
        writer.writerow(
            [
                line["author"]["steamid"],
                line["author"]["playtime_forever"],
                line["author"]["playtime_last_two_weeks"],
                line["language"],
                review,
                line['timestamp_created'],
                line['votes_up'],
                line['votes_funny'],
                review_type
            ]
        )
    output_csv.close()
            
