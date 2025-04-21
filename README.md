### EX6 - Information Retrieval Using Vector Space Model in Python

### DATE: 21/04/2025

### AIM: To implement Information Retrieval Using Vector Space Model in Python.

### Description: 

<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:

1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:

```
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
from tabulate import tabulate

nltk.download('punkt')
nltk.download('stopwords')

# Sample documents
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}

# Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
    return " ".join(tokens)

preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

# Vectorizers
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(preprocessed_docs.values())

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

terms = tfidf_vectorizer.get_feature_names_out()

# Term Frequency Table
print("\n--- Term Frequencies (TF) ---\n")
tf_table = count_matrix.toarray()
print(tabulate([["Doc ID"] + list(terms)] + [[list(preprocessed_docs.keys())[i]] + list(row) for i, row in enumerate(tf_table)], headers="firstrow", tablefmt="grid"))

# Document Frequency (DF) and IDF Table
df = np.sum(count_matrix.toarray() > 0, axis=0)
idf = tfidf_vectorizer.idf_

df_idf_table = []
for i, term in enumerate(terms):
    df_idf_table.append([term, df[i], round(idf[i], 4)])

print("\n--- Document Frequency (DF) and Inverse Document Frequency (IDF) ---\n")
print(tabulate(df_idf_table, headers=["Term", "Document Frequency (DF)", "Inverse Document Frequency (IDF)"], tablefmt="grid"))

# TF-IDF Table
print("\n--- TF-IDF Weights ---\n")
tfidf_table = tfidf_matrix.toarray()
print(tabulate([["Doc ID"] + list(terms)] + [[list(preprocessed_docs.keys())[i]] + list(map(lambda x: round(x, 4), row)) for i, row in enumerate(tfidf_table)], headers="firstrow", tablefmt="grid"))

# Manual Cosine Similarity calculation
def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0.0
    return dot_product, norm_vec1, norm_vec2, similarity

# Search function
def search(query, tfidf_matrix, tfidf_vectorizer):
    preprocessed_query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([preprocessed_query]).toarray()[0]
    results = []

    for idx, doc_vector in enumerate(tfidf_matrix.toarray()):
        doc_id = list(preprocessed_docs.keys())[idx]
        doc_text = documents[doc_id]
        dot, norm_q, norm_d, sim = cosine_similarity_manual(query_vector, doc_vector)
        results.append([doc_id, doc_text, round(dot, 4), round(norm_q, 4), round(norm_d, 4), round(sim, 4)])
    
    results.sort(key=lambda x: x[5], reverse=True)
    return results, query_vector

# Input from user
query = input("\nEnter your query: ")

# Perform search
results_table, query_vector = search(query, tfidf_matrix, tfidf_vectorizer)

# Display Cosine Similarity Table
print("\n--- Search Results and Cosine Similarity ---\n")
headers = ["Doc ID", "Document", "Dot Product", "Query Magnitude", "Doc Magnitude", "Cosine Similarity"]
print(tabulate(results_table, headers=headers, tablefmt="grid"))

# Display Ranking
print("\n--- Ranked Documents ---\n")
ranked_docs = []
for idx, res in enumerate(results_table, start=1):
    ranked_docs.append([idx, res[0], res[1], res[5]])

print(tabulate(ranked_docs, headers=["Rank", "Document ID", "Document Text", "Cosine Similarity"], tablefmt="grid"))

# Highest score
highest_score = max(row[5] for row in results_table)
print(f"\nThe highest rank cosine score is: {highest_score}")

# Display Query TF-IDF Weights
print("\n--- Query TF-IDF Weights ---\n")
query_weights = [(terms[i], round(query_vector[i], 4)) for i in range(len(terms)) if query_vector[i] > 0]
print(tabulate(query_weights, headers=["Term", "Query TF-IDF Weight"], tablefmt="grid"))
```


### Output:

![Screenshot 2025-04-21 191005](https://github.com/user-attachments/assets/1b7d91e8-1c77-4698-b488-32a8ad64b199)
![Screenshot 2025-04-21 191014](https://github.com/user-attachments/assets/1f52c137-7146-4dab-a54b-e48cc0ddc348)


### Result:

Thus the, Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, TF-IDF scores, and performing similarity calculations between queries and documents is executed successfully.
