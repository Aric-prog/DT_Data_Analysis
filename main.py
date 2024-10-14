import pandas as pd
import matplotlib.pyplot as plt

from models.playtime import basic_playtime_stats, playtime
from models.text import basic_review_text_stats, text_length
from models.upvotes import basic_upvote_stats, upvotes

plt.rcParams['figure.facecolor'] = '#EDDACF'
plt.rcParams['axes.facecolor'] = '#EDDACF'
def main():
    data = pd.read_csv('./data/dawntrail_reviews.csv')
    data['review_type'] = data['review_type'].map({'negative': 0, 'positive': 1})

    basic_playtime_stats(data.copy())
    basic_review_text_stats(data.copy())
    basic_upvote_stats(data.copy())

    text_length(data.copy())
    playtime(data.copy())
    upvotes(data.copy())
    

if __name__ == '__main__':
    main()