#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:57:42 2024

@author: orane
"""

import glob
import os

"""
Ce script calcule la distance entre les listes de voisins générées
précedemment. Il affiche la distance pour chaque liste en comparant les listes
de 3 types de vecteurs ainsi que la moyenne
"""

def load_neighbors(path):
    with open(path, "r") as file :
        neighbors = [line.split(",")[0] for line in file.readlines()]
    return neighbors

def compute_variation(lst_1, lst_2):
    """La variation est égale à 1 - l'intersection des deux 
       listes de voisins / le nombre de voisins
    """
    intersect = list(set(lst_1) & set(lst_2))
    var = 1 - len(intersect)/len(lst_1)
    return var

def load_files(path):
    return glob.glob(path)
def load_directories():
    return [folder for folder in glob.glob("../Nearest_neighbors/word_2_10/*") if os.path.isdir(folder)]


directories = load_directories()


for i in range(len(directories)):
    for j in range(i + 1, len(directories)):
        current_dir = directories[i]
        folder = directories[j]
        lst_var = []
       # print(f"Comparison {current_dir} & {folder}")
        name_current_dir = current_dir.split("/")[3]
        name_folder = folder.split("/")[3]
        file_current = load_files(f"{current_dir}/*.txt")
        file_comp = load_files(f"{folder}/*.txt")
        for k, file in enumerate(file_current):
            word = os.path.basename(file).split("_")[0]
            word_comp = os.path.basename(file_comp[k]).split("_")[0]
            neighbors_current = load_neighbors(file)
            neighbors_comp = load_neighbors(file_comp[k])
            var = compute_variation(neighbors_current, neighbors_comp)
            lst_var.append(var)
            #print(f"For {word} and {word_comp} variation = {var}")
        print(f"For {name_current_dir} and {name_folder} Moyenne : {sum(lst_var) / len(lst_var)}")


