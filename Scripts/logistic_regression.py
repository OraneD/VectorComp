#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 00:46:33 2024

@author: orane
"""
"""
Ce script lance un modèle de Régression Logistique pour procéder à une tâche
de classification du corpus.
Tous les vecteurs générés précédemment sont testés.
Le script afficher le rapport pour chaque type de modèle.
"""

from tokenize_and_clean import load_corpus, tokenize
from get_nearest_neighbors import load_vectors
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import glob

corpus = tokenize(load_corpus("../Corpus/*/*.txt"))
vectors = load_vectors("../Vectors/pca_vectors_100.txt")


def load_y(path):
    corpus = glob.glob(path)
    lst_y = [file.split("/")[2] for file in corpus]
    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(lst_y)
    encoded_y = encoded_y.ravel()
    class_names = label_encoder.classes_
    return encoded_y, class_names


def document_to_vector(doc, word_vectors):
    words = doc.split()
    if words:
        doc_vector = np.mean([word_vectors.get(word, np.zeros(len(next(iter(word_vectors.values()))))) for word in words], axis=0)
    else:
        doc_vector = np.zeros(len(next(iter(word_vectors.values()))))
    return doc_vector

def main():
    corpus = tokenize(load_corpus("../Corpus/*/*.txt"))
    for vector_file in glob.glob("../Vectors/*") :
        model_name = vector_file.split("/")[-1]
        vectors = load_vectors(vector_file)
        vectorized_corpus = [document_to_vector(doc, vectors)  for doc in corpus]
        y, class_names = load_y("../Corpus/*/*.txt")    
        X_train, X_test, y_train, y_test = train_test_split(vectorized_corpus, y, test_size=0.2, random_state=42)
        y_train = y_train.ravel()
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print()
        print(f"---------Classification Report for {model_name}------------")
        print(classification_report(y_test, predictions, target_names=class_names))

if __name__=="__main__":
    main()