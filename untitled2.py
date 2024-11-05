import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_idle_time(file_path, id_column_index=1, status_column_index=4, threshold=1, bins=10):
    """
    Analyse le temps d'attente à l'arrêt pour chaque agent en comptant le nombre d'occurrences
    où le statut est inférieur au seuil donné. Affiche le temps d'attente moyen et un histogramme
    de la distribution des temps d'arrêt.
    
    Paramètres :
    - file_path (str) : Chemin vers le fichier .csv.
    - id_column_index (int) : Index de la colonne contenant les identifiants d'agent.
    - status_column_index (int) : Index de la colonne contenant les valeurs de statut.
    - threshold (float) : Seuil en dessous duquel l'agent est considéré à l'arrêt (par défaut 1).
    - bins (int) : Nombre de bacs pour l'histogramme.
    """
    # Charger le fichier CSV
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV: {e}")
        return
    
    # Vérifier que le fichier a suffisamment de colonnes
    if data.shape[1] <= max(id_column_index, status_column_index):
        print("Erreur : L'index des colonnes dépasse le nombre de colonnes dans le fichier.")
        return
    
    # Extraire les colonnes pertinentes
    agent_ids = data.iloc[:, id_column_index]
    status_values = data.iloc[:, status_column_index]
    
    # Filtrer les lignes où l'agent est à l'arrêt (valeur < threshold)
    idle_data = data[status_values < threshold]
    
    # Compter les occurrences d'arrêt pour chaque agent
    idle_counts = idle_data.groupby(agent_ids).size()
    
    # Eliminer les arrets trop longs
    idle_counts =  idle_counts[ idle_counts > 100]    
    idle_counts =  idle_counts[ idle_counts < 1000]

    
    # Calculer le temps d'attente moyen à l'arrêt
    average_idle_time = idle_counts.mean()
    print(f"Temps d'attente à l'arrêt moyen (en nombre d'occurrences) : {average_idle_time:.2f}")
    

    
    # Afficher un histogramme de la distribution des temps d'attente à l'arrêt
    plt.figure(figsize=(10, 6))
    plt.hist(idle_counts, bins=bins, edgecolor='black')
    plt.title("Distribution du temps d'attente à l'arrêt par agent")
    plt.xlabel("Nombre d'occurrences d'arrêt")
    plt.yscale('log')
    plt.ylabel("Nombre d'agents")
    plt.grid(axis='y', alpha=0.75)
    
    X = np.linspace(100,999,1000)
    plt.plot(X, 200 * np.exp(-X / 200.67))
    
    plt.show()

# Exemple d'utilisation :
analyze_idle_time('file.csv')
