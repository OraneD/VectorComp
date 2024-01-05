#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 03:51:08 2024

@author: orane
"""

from tokenize_and_clean import load_corpus, tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

corpus = tokenize(load_corpus("../Corpus/*.txt"))



def get_simple_vectors(corpus):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    vectors_array = vectors.toarray()
    df = pd.DataFrame(data=vectors_array, columns = vectorizer.get_feature_names_out())
    print(df)

def reduce_dim(vectors):
    pass

def get_W2V_vectors(corpus):
    pass