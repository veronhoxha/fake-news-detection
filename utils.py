## UTILS ##
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch



def count_occurance(df, column):
    count = df[column].value_counts()

    top_n = count[:5]
    bottom_n = count[-5:]

    combined = pd.concat([top_n, bottom_n])
    
    colors = ['lightgreen' if x > combined.median() else 'red' for x in combined.values]
    
    sns.barplot(x=combined.index, y=combined.values, hue=combined.index, palette=colors, legend=False)
    plt.xticks(rotation=90)
    plt.xlabel(column.capitalize())
    plt.ylabel("Count")
    plt.title(f"Count of {column.capitalize()}")
    plt.show()


# removing punctation
def remove_special_chrs(text):
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    #text = re.sub("[^a-zA-Z]", " ", text)
    return text


def remove_web_addresses(text):
    text = text.replace(">", " ").replace("<", " ")
    # removing html tags if they exist
    WEB_RE = re.compile(r'(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})(\.[a-zA-Z0-9]{2,})?')

    return WEB_RE.sub('<URL>', text)