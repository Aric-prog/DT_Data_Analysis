import pandas as pd
import matplotlib.pyplot as plt

def get_neg_and_pos_mean_median(df: pd.DataFrame, column):
    return df.groupby('review_type').agg({column: ['mean', 'median']})

def plot_dataframe(df: pd.DataFrame, y_label, title, file_location=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(df.index))
    width = 0.35
    
    ax.bar([i - width / 2 for i in x], df['mean'], width, label='Mean', color='#9F715D')
    ax.bar([i + width / 2 for i in x], df['median'], width, label='Median', color='#5d8b9f')
    ax.set_facecolor('#EDDACF')
    fig.patch.set_facecolor('#EDDACF')
    
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.legend()
    
    plt.tight_layout()
    if file_location:
        plt.savefig(f"./figure/{file_location}")
    else:
        plt.show()
    plt.close("all")
    
def get_mean_and_median(df: pd.DataFrame):
    return df.mean(), df.median()

# def plot(positive_mean, positive_median, negative_mean, negative_median):
