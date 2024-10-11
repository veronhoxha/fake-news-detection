# Fake News Detection - Mini Project

Group members:
 - Arman Ul Alam (arua@itu.dk)
 - Gino Franco Fazzi (gifa@itu.dk)
 - Veron Hoxha (veho@itu.dk)

## Introduction
This repository hosts all the necessary resources for the Advanced Machine Learning mini-project ``"Fake News Detection"``.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
  - [Code](#code)
  - [Data](#data)
  - [Additional Folders](#additional-folders)
- [Project Goals](#project-goals)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)

## Project Structure

### Code
- `notebook.ipynb`: Main Jupyter Notebook where all the analyses are performed.
- `utils.py`: Miscelaneous and custom-made functions used in `notebook.ipynb`.

### Data
Located in the **"data"** folder:
- Directory containing the train data (X_train.csv, y_train.csv), test data (X_test.csv, y_test.csv), and the predictions from our model on the final test set (predictions.csv).

### Additional Folders
- `"models"`: Contains our best model saved in bin file type.
- `"literature"`: Contains the paper/litertature used for the reduced overview of this task.
- `"presentation"`: Contains the PowerPoint presentation of the project.

## Project Goals

- **Clarify and reflect on the definition of the term “fake news”, which may vary among databases, sometimes non-binary.**

  To first be able to detect and discriminate fake news, we need a somewhat clear definition of what constitutes fake news. This is far from trivial, as we will see now.
  Wikipedia defines Fake news (or information disorder) as "false or misleading information (misinformation, including disinformation, propaganda, and hoaxes) presented as news". This first definition is quite vague and abstract, and it is far from complete. As an example, would inaccurate data used in a news classified as fake news?

  Continuing with the Wikipedia explanation, we encounter one of the goals of fake news: "often has the aim of damaging the reputation of a person or entity, or making money through advertising revenue. (...)Further, disinformation involves spreading false information with harmful intent and is sometimes generated and propagated by hostile foreign actors, particularly during elections. In some definitions, fake news includes satirical articles misinterpreted as genuine, and articles that employ sensationalist or clickbait headlines that are not supported in the text. Because of this diversity of types of false news, researchers are beginning to favour information disorder as a more neutral and informative term".

  We have now a somehow more clear picture: the tendency is to put fake news as a subgenre in the more neutral and nuanced "information disorder", with the main difference of having a harfmul purpose.
  Then, the taks of fake news detection becomes not only an objectiva binary task between something being factually correct or incorrect, but also analyizing the purpose of the originator or distributor of the news. The latter task is inherently difficult, since it requires more context than just the written text. 

- **Research, where the data comes from and inspect the data: what are the labels, sources, and authors? Is there a person, source or topic which is over- or under-represented?**

  The data we are analyzing is sourced from the Fake News Kaggle competition (https://www.kaggle.com/competitions/fake-news/code?competitionId=8317&sortBy=voteCount&excludeNonAccessedDatasources=true). It is divided into training and test sets, with the training set fully labeled. The test set includes 4,160 entries and the training set includes 16,640 entries across four columns: 

  1. **title** - the tile of the news
  2. **author** - the author of the news
  3. **text** - the text of the news
  4. **label** - a label that tells whether the news is real or fake
  
  The labels in the training data are integers, where '0' indicates REAL news and '1' represents FAKE news, with no missing values - there are 8,385 instances of FAKE news and 8,255 of REAL news.

  The **"title"** column is a string type with 458 null entries. Analysis shows a generally balanced title frequency, although some, like **"Will Barack Obama Delay Or Suspend The Election If Hillary Is Forced Out By The New FBI Email Investigation?",** appear multiple times (four occurrences), suggesting over-representation. In contrast, titles such as **"Heseltine strangled dog as part of Thatcher cabinet initiation ceremony"** are less common, recorded only once, highlighting under-representation.

  The **"author"** column is a string type with 1,574 null entries. Our analysis highlights a significant imbalance in **"author"** frequency within our training set. **"Pam Key"** is notably prevalent, appearing 193 times, suggesting over-representation. Conversely, **"Peter Koenig"** and several others are mentioned only once, indicating under-representation. This disparity underscores the need for a more balanced author distribution in our dataset.

  The **"text"** column is a string type with 30 null entries. The most repeated text is found 58 times, indicating over-representation, while several texts appear only once, showing under-representation. Additionally, our keyword analysis reveals that political terms such as **"trump," "clinton," "president," "government," "obama," and "state"** are the most frequent, emphasizing a strong political focus in our news dataset.

- **Offer a reduced overview of the literature on the most used approaches for this task.**

  Our review focuses on the literature detailed in **A Comprehensive Review on Fake News Detection With Deep Learning** accessible via [A Comprehensive Review on Fake News Detection With Deep Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9620068).
    
  This paper examines the use of the same Kaggle dataset that our project employs.
  
  Initially, the paper outlines several preprocessing steps like tokenization, stemming, and lemmatization that could enhance the classifier performance. However, many researchers skip these steps with the reason that they still achieve robust results. In our project, we incorporated preprocessing methods such as punctuation removal, text lowercasing, HTML tag removal, and tokenization to potentially boost performance.
  
  The literature gives details on why using Deep Learning (DL) models is better than Machine Learning (ML) models when it comes to tasks related to fake news detection. The paper claims that DL models take the advantage when it comes to:

    - automated feature extraction, lightly dependent on data pre-processing, 
    - ability to extract high-dimensional features, and 
    - better accuracy.

  The table below, extracted from the paper, shows the accuracy of DL-based studies along with used method and NLP techniques applied to our dataset:

    | Method                  | NLP Techniques              | Accuarcy         |
    |-------------------------|:---------------------------:| ----------------:|
    | CNN                     | TF-IDF                      | 98.3%            |
    | Deep CNN                | GloVe                       | 98.36%           |
    | CNN                     | Tensorflow embedding layer  | 96%              |
    | CNN + LSTM              | GloVE                       | 94.71%           |
    | Bi-directional LSTM-RNN | GloVe                       | 98.75%           |
    | Passive aggressive      | TF-IDF                      | 83.8%            |
    | FakeBert                | GloVe, `BERT`               | `98.80%`         |

    These results show that various methods employing different NLP techniques achieve significantly high accuracy. Notably, the Passive Aggressive method with TF-IDF achieves an accuracy lower than the other ones with 83.8%, which is still considered good. Despite the small differences in accuracy among the reviewed models, we chose BERT due to its robust ability to understand and incorporate context, which is crucial for identifying subtle nuances in fake news. BERT excels because it learns from a diverse array of texts and then adapts specifically to fake news detection. BERT operates in two stages: pre-training and fine-tuning, capturing contextual meanings in text effectively.

## Installation
Ensure Python 3.10.11 is installed and then run:
- `pip install -r requirements.txt`

## Usage
To run the notebook and recreate results:
1. Run `notebook.ipynb` using the "Run All" feature in VS Code or another IDE.
