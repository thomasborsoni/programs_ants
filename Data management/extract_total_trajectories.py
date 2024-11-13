import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_complete_trajectories(file_path, id_column_index=1, time_column_index=0, x_position_column_index=2, x_start_threshold=5, x_end_threshold=95):
    """
    Extrait les ID des agents ayant des trajectoires complètes, définies par :
    - Position x inférieure à `x_start_threshold` lors de leur première apparition.
    - Position x supérieure à `x_end_threshold` lors de leur dernière apparition.
    
    Paramètres :
    - file_path (str) : Chemin vers le fichier .csv.
    - id_column_index (int) : Index de la colonne contenant les identifiants d'agent.
    - time_column_index (int) : Index de la colonne contenant l'instant (ou le temps).
    - x_position_column_index (int) : Index de la colonne contenant la position x.
    - x_start_threshold (float) : Seuil de position x pour la première apparition (par défaut 5).
    - x_end_threshold (float) : Seuil de position x pour la dernière apparition (par défaut 95).
    
    Retourne :
    - complete_trajectories (list) : Liste des ID d'agents ayant des trajectoires complètes.
    """
    # Charger le fichier CSV
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV: {e}")
        return
    
    # Extraire les colonnes pertinentes
    agent_ids = data.iloc[:, id_column_index]
    times = data.iloc[:, time_column_index]
    x_positions = data.iloc[:, x_position_column_index]
    v_mag = data.iloc[:, 4]
    
    # Créer un DataFrame temporaire
    temp_df = pd.DataFrame({'time': times, 'agent_id': agent_ids, 'x_position': x_positions, 'speed': v_mag})
    
    # Identifier le premier et le dernier enregistrement pour chaque agent
    first_positions = temp_df.sort_values(by='time').groupby('agent_id').first()
    last_positions = temp_df.sort_values(by='time').groupby('agent_id').last()
    
    # Filtrer les agents ayant x < x_start_threshold au premier instant et x > x_end_threshold au dernier instant
    complete_trajectories_ids = first_positions[(first_positions['x_position'] < x_start_threshold) & 
                                                (last_positions['x_position'] > x_end_threshold)].index.tolist()

    
    complete_trajectories_df = temp_df[temp_df['agent_id'].isin(complete_trajectories_ids)]
    
    existence_time = complete_trajectories_df.groupby(complete_trajectories_df.iloc[:, 1]).size()
    average_velocity = complete_trajectories_df.groupby(complete_trajectories_df.iloc[:, 1])['speed'].mean()
    
    idle_data = complete_trajectories_df[complete_trajectories_df['speed']<1]
    idle_counts = idle_data.groupby(complete_trajectories_df.iloc[:, 1]).size()
    
    
    # Plot the histogram of velocities for complete trajectories
 #   plt.figure(figsize=(10, 6))
   # plt.hist(complete_trajectories_df['speed'].dropna(), bins=100, edgecolor='black')
  #  plt.hist(idle_counts[idle_counts<300], bins=100, edgecolor='black')
   # plt.xscale('log')
  #  plt.yscale('log')
  #  plt.xlim([-1,300])
  #  plt.grid(axis='y', alpha=0.75)
  #  plt.show()
    
    plt.figure(figsize=(10, 6))
   # plt.hist(complete_trajectories_df['speed'].dropna(), bins=100, edgecolor='black')
    plt.hist(2500 / existence_time[existence_time < 300], bins=100, edgecolor='black', alpha=.7, label='Avg. crossing vel.')
    plt.hist(average_velocity[existence_time < 300], bins=100, edgecolor='black', alpha=.3, label='Avg. instant. vel.')
    plt.yscale('log')
    plt.title('Histogram of average crossing and instantaneous velocity')
    plt.xlabel('Vel. magnitude')
    plt.ylabel('Occurences')
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.savefig('azertyuiop.pdf')
    plt.show()
    

  #  plt.figure(figsize=(10, 6))
   # plt.hist(complete_trajectories_df['speed'].dropna(), bins=100, edgecolor='black')
  #  plt.hist((average_velocity * existence_time/25)[existence_time<300], bins=100, edgecolor='black')
   # plt.xscale('log')
   # plt.yscale('log')
  #  plt.xlim([-1,30])
    #plt.grid(axis='y', alpha=0.75)
    #plt.show()
    
    return 0

# Exemple d'utilisation :
    
file_path = "/Users/thomasborsoni/Desktop/Post-doc/Projet fourmis/Programmes/Data management/file.csv"

#complete_traj = extract_complete_trajectories(file_path, x_start_threshold=np.infty, x_end_threshold=-np.infty)
complete_traj = extract_complete_trajectories(file_path, x_start_threshold=5, x_end_threshold=95)











