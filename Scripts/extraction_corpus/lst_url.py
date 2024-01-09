#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:30:10 2024

@author: orane
"""
"""
Ce script permet de récupérer les listes d'URL pour chaque résumé
en allant chercher les identifiants des thèses dans le les fichers xml
qui correspondent aux recherches sur le site thèses.fr.
Il faut changer les chemins pour chaque discipline
"""

import os
from pathlib import Path
from xml.etree import ElementTree as et


print(os.getcwd())



with open(Path(os.getcwd() + "/Litterature/url_litterature.txt"), "w") as file : 
    xml = et.parse(os.getcwd() + "/Litterature/litterature_2010_2023.xml")
    root = xml.getroot()
    nb = 0
    for string in root.findall(".//str") :
        if string.attrib == {"name" : "num"} :
           file.write("https://www.theses.fr/" + string.text + ".xml" + "\n")
           nb += 1
        #if nb == 200 :
            #break
