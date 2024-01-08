#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 03:51:08 2024

@author: orane
"""

from tokenize_and_clean import load_corpus, tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from collections import Counter
from itertools import product
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

"""
Ce script permet d'obtenir les 3 types de vecteurs :
    - Avec une matrice PPMI
    - Avec réduction de dimensionnalité (PCA) sur les vecteurs PPMI
    - Avec Word2Vec
Les vecteurs sont générés et copiés dans des fichiers textes placés dans le dossier ../Vectors
"""



def load_stopwords(path):
    with open(path, "r") as f :
        stopwords = f.read().split("\n")
    return stopwords


def get_PPMI_vectors(corpus,file_path, stopwords,min_df=5, max_df=0.50, max_features=None):
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, stop_words=stopwords)
    X = vectorizer.fit_transform(corpus)
    sum_words = X.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    with open("../utils/vocabulary.txt", "w") as file:
        for word in words_freq :
            file.write(f"{word[0]}, {word[1]} \n")
    Xc = (X.T * X)  # Matrice de cooccurrence
    Xc.setdiag(0)   # Suppression de la diagonale (fréquence du mot avec lui-même)
    feature_names = vectorizer.get_feature_names_out()
    cooccurrence_matrix = Xc.toarray()
    # Calcul des probabilités marginales
    total_cooccurrences = cooccurrence_matrix.sum()
    word_probabilities = cooccurrence_matrix.sum(axis=1) / total_cooccurrences
    # Calcul de la matrice de probabilités conjointes
    joint_probabilities = cooccurrence_matrix / total_cooccurrences
    # Calcul de la PPMI
    ppmi_matrix = np.maximum(np.log2(joint_probabilities / np.outer(word_probabilities, word_probabilities)),0)
    df_ppmi = pd.DataFrame(ppmi_matrix, index=feature_names, columns=feature_names)
    with open(file_path, 'w', encoding='utf-8') as file:
        for word in df_ppmi.index:
            vector_str = ",".join(map(str, df_ppmi.loc[word]))
            file.write(f"{word},{vector_str}\n")
    print("-------PPMI vectors------")
    print(df_ppmi)
    return ppmi_matrix, vectorizer


def reduce_dim_with_pca(vectors, vectorizer, nb_dim=10):
    pca = PCA(n_components=nb_dim)
    pca_results = pca.fit_transform(vectors)
    words = vectorizer.get_feature_names_out()
    word_pca = pd.DataFrame(pca_results, columns=[f"PCA{i+1}" for i in range(nb_dim)])
    word_pca.index = words
    with open(f"../Vectors/pca_vectors_{nb_dim}.txt", "w") as file:
        for word, values in word_pca.iterrows():
            line = f"{word}," + ",".join(map(str, values))
            file.write(line + "\n")
    print()
    print("---------PCA Vectors----------")
    print(word_pca)
    print(f"{len(word_pca)} vectors of {nb_dim} dimensions")
    return word_pca

def get_W2V_vectors(corpus, vectorizer, file_path, vector_size=100, window=5, min_count=1, workers=4):
    #Word2Vec ne prend pas la même forme de corpus en entrée que CountVectorizer : 
    #Il prend une liste de liste de mot alors que la forme initiale du corpus est une liste de string
    corpus_W2V = []
    for doc in corpus :
        corpus_W2V.append(doc.split())
    # Entraînement du modèle Word2Vec
    model = Word2Vec(corpus_W2V, vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=1, epochs=50)
    # On récupère les mots qui constituent nos précédents vecteurs
    unique_words = list(vectorizer.get_feature_names_out())
    # Pour chaque mot, on extrait son vecteur avec le modèle que l'on vient d'entraîner
    word_vectors = {}
    for word in unique_words:
        if word in model.wv:
            word_vectors[word] = model.wv[word]
        else:
            pass
    with open(f"{file_path}_{vector_size}.txt", "w") as file:
        for word, vector in word_vectors.items():
            vector_str = ",".join(map(str, vector))
            file.write(f"{word},{vector_str}\n")
    
    print()
    print("------------W2V Vectors------------")
    print(f"{len(word_vectors)} word vectors of {vector_size} dimension ")
    return word_vectors


def main():
    corpus = tokenize(load_corpus("../Corpus/*.txt"))
    fr_stop = load_stopwords("../utils/stop_words_fr.txt")
    ppmi_vectors, vectorizer = get_PPMI_vectors(corpus, "../Vectors/PPMI_vectors.txt",fr_stop, min_df=5, max_df=0.50, max_features=3500)
    reduce_dim_with_pca(ppmi_vectors, vectorizer, nb_dim=100)
    get_W2V_vectors(corpus, vectorizer, "../Vectors/W2V_vectors", vector_size=10, window=5, min_count=1, workers=4)

main()
'''   
Si jamais on veut les vecteurs TF-IDF..

def get_simple_vectors(corpus,file_path, min_df=1, max_df=1.0, max_features=None):
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, stop_words=fr_stop)
    vectors = vectorizer.fit_transform(corpus)
    tfidf_transformer = TfidfTransformer()
    tfidf_vectors = tfidf_transformer.fit_transform(vectors)
    vectors_array = tfidf_vectors.toarray()
    df = pd.DataFrame(data=vectors_array, columns=vectorizer.get_feature_names_out())
    write_vectors_to_txt(df,file_path)
    print("-------TD-IDF vectors------")
    print(df)
    return vectors_array, vectorizer
'''

