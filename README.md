# Keyword Extraction with TF-IDF

## Overview

This repository contains a Python script for extracting keywords from abstracts using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. The process involves cleaning and preprocessing text data, removing stopwords, and applying TF-IDF to identify significant terms.

## File Descriptions

- **finalfinal_file.csv**: The input CSV file containing abstracts and titles.
- **df_selected_with_keywords2.csv**: The output CSV file with extracted keywords for each abstract.
- **Keyword_Extraction.py**: The Python script for processing the data and extracting keywords.

## Dependencies

The script requires the following Python packages:
- pandas
- numpy
- re
- nltk
- scikit-learn

You can install the required packages using pip:

```bash
pip install pandas numpy nltk scikit-learn

## Instructions

Prepare Your Environment: Ensure you have Python and the required libraries installed.
Download the Data: Place the finalfinal_file.csv file in the same directory as the script.
Run the Script: Execute the script to process the data and extract keywords.
python Keyword_Extraction.py
Check the Output: The keywords will be saved in the df_selected_with_keywords2.csv file.

