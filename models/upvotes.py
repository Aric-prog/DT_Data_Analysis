import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from nltk.tokenize import word_tokenize
import nltk
import seaborn as sns

from models.text import text_to_length
from models.util import get_neg_and_pos_mean_median, plot_dataframe

def basic_upvote_stats(df: pd.DataFrame):
    df.review = df.apply(text_to_length, axis=1)
    df = df[df.review > 0]

    df = get_neg_and_pos_mean_median(df, 'votes_up')
    df = df.votes_up

    plot_dataframe(df, 'Upvotes', 'Mean and Median Upvotes by Review Type', "upvotes")

# Does upvotes correlate with review sentiment?
def upvotes(df: pd.DataFrame):
    df.review = df.apply(text_to_length, axis=1)
    df = df[df.review > 0]
    df = df[df.votes_up > 0]

    X = df[['votes_up']]
    y = df.review_type

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

    # Partial dependence graph, not very relevant

    # fig, ax = plt.subplots(figsize=(10, 6))

    # ax.set_facecolor('#EDDACF')
    # fig.patch.set_facecolor('#EDDACF')
    # PartialDependenceDisplay.from_estimator(
    #     model,  
    #     X,              
    #     ['votes_up'],   
    #     kind='average', 
    #     ax=ax,
    #     line_kw={'color': 'blue'},
    # )
    # ax.set_xlabel('Upvotes')
    # ax.set_ylabel('Partial Dependence')
    # ax.set_title('Partial Dependence of Review Sentiment on Upvotes')
    # plt.show()

    ax = sns.regplot(x='votes_up', y='review_type', data=df, logistic=True, ci=None, 
                scatter_kws={'alpha': 0.1, 'color':'#9F715D'}, 
                line_kws={'color':'#5d8b9f'}
        )
    ax.set_ylabel('Probability of Positive Review')
    ax.set_xlabel('Upvote count')
    plt.savefig("./figure/upvote_regression")
    plt.close("all")


