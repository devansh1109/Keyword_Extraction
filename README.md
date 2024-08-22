Keyword Extraction from Abstracts
Overview
This project performs keyword extraction from a dataset of academic abstracts using Natural Language Processing (NLP) techniques. The code processes text data to clean, tokenize, and vectorize abstracts and then extracts the most relevant keywords using TF-IDF (Term Frequency-Inverse Document Frequency).

Prerequisites
Ensure you have the following Python packages installed:

pandas
numpy
re
nltk
sklearn
You can install the required packages using pip:

bash
Copy code
pip install pandas numpy nltk scikit-learn
File Descriptions
finalfinal_file.csv: The input CSV file containing the dataset with columns Title_y and abstract.
df_selected.csv: The CSV file saved after processing and cleaning the abstracts.
df_selected_with_keywords2.csv: The CSV file containing the abstracts along with the extracted keywords.
Steps and Code
1. Import Libraries
The code imports essential libraries for data manipulation and NLP:

python
Copy code
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
2. Load and Select Data
Load the CSV file and select relevant columns:

python
Copy code
df = pd.read_csv('finalfinal_file.csv')
columns = ['Title_y', 'abstract']
df_selected1 = df[columns]
3. Text Preprocessing
Convert text to lowercase, remove punctuation, and handle stopwords:

python
Copy code
df_selected1['abstract'] = df_selected1['abstract'].str.lower()

def remove_punctuations(text):
    pattern = r'[!"#$%&\'()*+,\-./:;<=>?@[\\]^_`{|}~]'
    return re.sub(pattern, '', text)

df_selected1['abstract'] = df_selected1['abstract'].apply(lambda x: remove_punctuations(str(x)) if pd.notnull(x) else x)

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    else:
        return text

df_selected1['abstract'] = df_selected1['abstract'].apply(remove_stopwords)
df_selected1['abstract'].replace('', np.nan, inplace=True)
4. TF-IDF Vectorization
Initialize TfidfVectorizer, fit and transform the data:

python
Copy code
vectorizer = TfidfVectorizer(
    stop_words='english',
    token_pattern=r'\b[a-zA-Z]{4,}\b',
    ngram_range=(1, 3),
    max_df=0.1
)

tfidf_matrix = vectorizer.fit_transform(df_selected1['abstract'])
feature_names = vectorizer.get_feature_names_out()
5. Extract Keywords
Define a function to extract top N keywords for each abstract:

python
Copy code
def extract_top_keywords(row, top_n=5):
    try:
        row_index = row.name
        tfidf_scores = tfidf_matrix[row_index].toarray()[0]
        tfidf_df = pd.DataFrame({'word': feature_names, 'tfidf': tfidf_scores})
        tfidf_df = tfidf_df.sort_values(by='tfidf', ascending=False)
        top_keywords = tfidf_df.head(top_n)['word'].values
        return ', '.join(top_keywords)
    except IndexError:
        return None

df_selected1['keywords'] = df_selected1.apply(extract_top_keywords, axis=1)
6. Save Results
Save the final DataFrame to a CSV file:

python
Copy code
df_selected1.to_csv('df_selected_with_keywords2.csv', index=False)
Usage
Place the finalfinal_file.csv in the working directory.
Run the script to process the data and extract keywords.
The processed data will be saved to df_selected_with_keywords2.csv.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to adjust the README file according to your specific needs and project details.
