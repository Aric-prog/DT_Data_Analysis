import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from sklearn.inspection import PartialDependenceDisplay
import nltk
import seaborn as sns
import numpy as np
from models.text import text_to_length
from models.util import get_neg_and_pos_mean_median, plot_dataframe

def basic_playtime_stats(df: pd.DataFrame):
    df = df[df.playtime_forever > 0]
    df_two_weeks = df.copy()

    df = get_neg_and_pos_mean_median(df, 'playtime_forever')
    df = df.playtime_forever // 60.0
    
    plot_dataframe(df, 'Playtime (hours)', 'Mean and Median Playtime by Review Type', "playtime_mean")

    df_two_weeks = get_neg_and_pos_mean_median(df_two_weeks, 'playtime_two_weeks')
    df_two_weeks = df_two_weeks.playtime_two_weeks // 60.0
    
    plot_dataframe(df_two_weeks, 'Playtime (hours)', 'Mean and Median Playtime of Last Two Weeks  by Review Type', "playtime_two_weeks_mean")
    
# Does playtime and playtime on last two weeks correlate with reviews?
def playtime(df: pd.DataFrame):
    df = df[df.playtime_forever > 0]
    df.playtime_forever = df.playtime_forever // 60.0
    X = df[['playtime_forever']]
    y = df.review_type

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(10, 6))

    sns.regplot(x='playtime_forever', y='review_type', data=df, logistic=True, ci=None, 
                scatter_kws={'alpha': 0.1, 'color':'#9F715D'}, 
                line_kws={'color':'#5d8b9f'})
    plt.title('Logistic Regression: Playtime vs Review Sentiment')
    plt.xlabel('Playtime (in hours)')
    # plt.ylabel('Probability of Positive Review')
    plt.savefig("./figure/playtime_regression")
    plt.close("all")



