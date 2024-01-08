#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:34:22 2024

@author: orane
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def load_simlex_fr(path):
    simlex_dict = {}
    with open(path, "r") as file :
        for i, line in enumerate([x.split(",")for x in file.readlines()]):
            if i == 0 :
                continue
            word_pair = line[1], line[2]
            scores = [int(x) for x in line[4:] if x != ""]
            average_score = sum(scores) / len(scores)
            simlex_dict[word_pair] = average_score
    return simlex_dict
    

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
    reduced_vectors = load_vectors("../Vectors/pca_vectors_10.txt")
    W2V_vectors = load_vectors("../Vectors/W2V_vectors_100.txt")
    vectors = [PPMI_vectors, reduced_vectors,W2V_vectors]
    return vectors

def cosine_similarity_for_pairs(vectors_dict, simlex_dict):
    similarity_results = {}
    pair = 0
    for word_pair, _ in simlex_dict.items():
        word1, word2 = word_pair
        # Vérifier si les deux mots sont dans les vecteurs de mots
        if word1 in vectors_dict and word2 in vectors_dict:
            pair += 1
            # Obtenir les vecteurs pour les deux mots
            vector1 = vectors_dict[word1].reshape(1, -1)
            vector2 = vectors_dict[word2].reshape(1, -1)
            # Calculer la similarité cosinus et stocker le résultat
            similarity = cosine_similarity(vector1, vector2)[0][0]
            similarity_results[word_pair] = similarity
    missing_pairs = len(simlex_dict) - pair
    return similarity_results



from scipy.stats import spearmanr

def spearman_correlation(simlex_dict, vector_similarities):
    simlex_scores = []
    vector_scores = []
    for word_pair, simlex_score in simlex_dict.items():
        if word_pair in vector_similarities and vector_similarities[word_pair] is not None:
            simlex_scores.append(simlex_score)
            vector_scores.append(vector_similarities[word_pair])
    correlation, p_value = spearmanr(simlex_scores, vector_scores)
    return correlation, p_value

simlex_dict = load_simlex_fr("../utils/FRA.csv")
vectors= load_vectors_file()
correlations = []
p_values = []
models = ['PPMI', 'PCA', 'W2V']
for vectors_dict in vectors :
    print("##################################################################")
    vector_similarities = cosine_similarity_for_pairs(vectors_dict,simlex_dict)
    correlation, p_value  = spearman_correlation(simlex_dict, vector_similarities)
    correlations.append(correlation)
    p_values.append(p_value)



fig, ax = plt.subplots()

barlist = plt.bar(models, correlations, alpha=0.7)
for i, p in enumerate(p_values):
    if p < 0.05:
        barlist[i].set_color('g')
    else:
        barlist[i].set_color('r')


for i in range(len(correlations)):
    plt.text(i, correlations[i], f'p-value={p_values[i]:.2f}', ha='center', va='bottom')
plt.xlabel('Models')
plt.ylabel('Spearman Correlation')

plt.show()

    
    

