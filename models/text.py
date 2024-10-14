import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from nltk.tokenize import word_tokenize
import nltk
import seaborn as sns

from models.util import get_neg_and_pos_mean_median, plot_dataframe

def text_to_length(row):
    return len(row['review'])

def basic_review_text_stats(df: pd.DataFrame):
    df.review = df.apply(text_to_length, axis=1)
    df = df[df.review > 0]
    df = df[df.language == "english"]

    df = get_neg_and_pos_mean_median(df, 'review')
    df = df.review

    plot_dataframe(df, 'Review character length', 'Mean and Median Review Length by Review Type (English only)', "review_length")

def text_length(df: pd.DataFrame):
    df.review = df.apply(text_to_length, axis=1)
    df = df[df.review > 0]
    X = df[['review']]
    y = df.review_type

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print(metrics.classification_report(y_test, y_pred))

    plt.figure(figsize=(10, 6))

    sns.regplot(x='review', y='review_type', data=df, logistic=True, ci=None, 
                scatter_kws={'alpha': 0.1, 'color':'#9F715D'}, 
                line_kws={'color':'#5d8b9f'})
    plt.title('Logistic Regression: Review length vs Sentiment')
    plt.xlabel('Review length')
    plt.ylabel('Probability of Positive Review')

    plt.savefig("./figure/review_length_regression")
    plt.close("all")
