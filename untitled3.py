import pandas as pd
import matplotlib.pyplot as plt

def analyze_simultaneous_agents(file_path, frame_column_index=0, agent_id_column_index=1, bins=30):
    """
    Analyse le nombre d'agents détectés simultanément dans chaque frame et affiche
    un histogramme de la distribution du nombre d'agents par frame.
    
    Paramètres :
    - file_path (str) : Chemin vers le fichier .csv.
    - frame_column_index (int) : Index de la colonne contenant les numéros de frame.
    - agent_id_column_index (int) : Index de la colonne contenant les identifiants d'agent.
    - bins (int) : Nombre de bacs pour l'histogramme.
    """
    # Charger le fichier CSV
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV: {e}")
        return
    
    # Vérifier que le fichier a suffisamment de colonnes
    if data.shape[1] <= max(frame_column_index, agent_id_column_index):
        print("Erreur : L'index des colonnes dépasse le nombre de colonnes dans le fichier.")
        return
    
    # Extraire les colonnes pertinentes
    frames = data.iloc[:, frame_column_index]
    agent_ids = data.iloc[:, agent_id_column_index]
    
    # Créer un DataFrame temporaire pour grouper les frames avec les agents
    temp_df = pd.DataFrame({'frame': frames, 'agent_id': agent_ids})
    
    # Compter le nombre d'agents uniques par frame
    agents_per_frame = temp_df.groupby('frame')['agent_id'].nunique()
    
    # Afficher un histogramme de la distribution du nombre d'agents par frame
    plt.figure(figsize=(10, 6))
    plt.hist(agents_per_frame, bins=bins, edgecolor='black')
    plt.title("Distribution du nombre d'agents détectés simultanément par frame")
    plt.xlabel("Nombre d'agents simultanés")
    plt.ylabel("Nombre de frames")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# Exemple d'utilisation :
analyze_simultaneous_agents('file.csv')
