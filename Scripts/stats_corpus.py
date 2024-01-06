#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 02:55:34 2024

@author: orane
"""

from tokenize_and_clean import load_corpus, tokenize

corpus = tokenize("../Corpus/*.txt")
average_length = sum(len(doc.split()) for doc in corpus) / len(corpus)
print(average_length)



