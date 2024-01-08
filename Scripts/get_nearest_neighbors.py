#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 12:52:31 2024

@author: orane
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

"""
Ce script récupère les 3 types de vecteur et calcule les 
K plus proches voisins de chaque mots pour chaque modèle.
Les K plus proches voisins pour chaque mot sont ensuite
copiés dans un fichier .txt qui porte le nom du mot pour lequel
le calcul est réalisé
"""

#Chargement de la liste de mot
def load_words():
    with open("../Nearest_neighbors/word_2.txt", "r") as file :
        words = [word.strip() for word in file.readlines()]
    return words


def load_vectors(file_path):
    vectors = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            split_line = line.strip().split(',')
            word = split_line[0]
            vector = np.array([float(x) for x in split_line[1:]])
            vectors[word] = vector
    return vectors

def load_vectors_file():
    PPMI_vectors = load_vectors("../Vectors/PPMI_vectors.txt")
    reduced_vectors = load_vectors("../Vectors/pca_vectors_100.txt")
    W2V_vectors = load_vectors("../Vectors/W2V_vectors_10.txt")
    vectors = [PPMI_vectors, reduced_vectors,W2V_vectors]
    return vectors

def find_nearest_neighbors(target_word, vectors, k=10):
    target_vector = vectors[target_word]
    similarities = {}
    for word, vector in vectors.items():
        if word != target_word:
            similarity = cosine_similarity([target_vector], [vector])[0][0]
            similarities[word] = similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:k]


def main():
    words = load_words()
    vectors_file = load_vectors_file()

    for i, vector in enumerate(vectors_file) :
        lst = ["ppmi", "pca_100", "W2V_10"]
        for word in words :
            with open(f"../Nearest_neighbors/neighbors_{lst[i]}/{word}_neighbors.txt", "w") as file :
                ten_neighbors = find_nearest_neighbors(word,vector,50) #Choix de K ici
                for neighbor in ten_neighbors :
                    file.write(f'{neighbor[0]},{neighbor[1]}\n')
                
if __name__=="__main__":
    main()



