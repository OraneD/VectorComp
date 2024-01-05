#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 03:36:29 2024

@author: orane
"""

import re
import glob

def load_corpus(path):
    lst_files = glob.glob(path)
    corpus =[]
    for file in lst_files :
        file_content = []
        with open(file, 'r') as f :
            text = f.read()
            corpus.append(text)
    return corpus

def tokenize(corpus):
    corpus = load_corpus("../Corpus/*/*.txt")
    return [re.sub("[^a-zA-Z\s]", "", text).lower() for text in corpus]


