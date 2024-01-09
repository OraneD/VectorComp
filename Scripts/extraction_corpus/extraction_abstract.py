#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:35:09 2024

@author: orane
"""
"""
Ce script prend en entrée les listes d'URL générés par le script lst_url.py
Il permet pour  une URL donnée, d'aller récupérer le résumé de la thèse considérée.
Il faut changer les chemins pour chaque discipline.
"""

import os
from xml.etree import ElementTree as et
import requests


print(os.getcwd())

nsmap = {"xml": "http://www.w3.org/XML/1998/namespace"}
nb = 0 

with open(os.getcwd() + "/Litterature/lt.txt", "r") as file :
    urls = file.readlines()
    for url in urls : 
        print(url)
        name = url.split("/")[-1].split(".")[0]
        tree = et.fromstring( requests.get(url.strip()).text)
        for child in tree.findall(".//{http://purl.org/dc/terms/}abstract[@xml:lang='fr']",namespaces=nsmap) :
            if child.text == "" or child.text == "Le résumé en français n'a pas été communiqué par l'auteur" or len(child.text) < 200 :
                print(name + "Pas de résumé")
            with open(os.getcwd() + "/Litterature/" + name + ".txt", "w") as resume :
                 nb += 1
                 print(str(nb) + " : "  + url)
                 resume.write(child.text)
        if nb == 200 :
            break
 
