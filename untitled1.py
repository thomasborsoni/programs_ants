#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:53:40 2024

@author: thomasborsoni
"""

import pandas as pd
import matplotlib.pyplot as plt

def count_values_below_threshold(file_path, column_index=4, threshold=1):
    """
    Lit un fichier CSV, extrait une colonne spécifique (par défaut la 5e),
    compte les valeurs inférieures à un seuil donné et trace un histogramme en échelle logarithmique.
    
    Paramètres :
    - file_path (str): Chemin vers le fichier .csv.
    - column_index (int): Index de la colonne à analyser (0 pour la première colonne, 4 pour la cinquième).
    - threshold (int ou float): Seuil pour compter les valeurs en dessous.
    - bins (int): Nombre de bacs pour l'histogramme (par défaut 50).
    - max_value (int ou float): Valeur maximale incluse dans l'histogramme (par défaut 50).
    """
    # Charger le fichier CSV
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV: {e}")
        return
    
    # Vérifier que le fichier a assez de colonnes
    if data.shape[1] <= column_index:
        print("Erreur : L'index de la colonne dépasse le nombre de colonnes dans le fichier.")
        return
    
    # Extraire les données de la colonne spécifiée
    column_data = data.iloc[:, column_index]
    column_label = data.columns[column_index]
    
    # Compter les valeurs inférieures au seuil
    count_below_threshold = (column_data < threshold).sum()
    print(f"Nombre de valeurs dans '{column_label}' en dessous de {threshold}: {count_below_threshold}")
    print(len(column_data))
    


# Exemple d'utilisation :
count_values_below_threshold('file.csv', threshold=.5)
