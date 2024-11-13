import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram_from_csv(file_path, column_index=4, bins=1000, max_value=50):
    """
    Reads a CSV file, extracts the specified column (default is the 5th column),
    filters out values above the specified max_value, and plots a histogram of the remaining values.
    
    Parameters:
    - file_path (str): Path to the .csv file.
    - column_index (int): Index of the column to plot (0-based, so 4 is the 5th column).
    - bins (int): Number of bins for the histogram (default is 50).
    - max_value (int or float): Maximum value threshold; values above this will be excluded.
    """
    # Load the CSV file into a DataFrame

    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Check if the file has enough columns
    if data.shape[1] <= column_index:
        print("Error: The specified column index exceeds the number of columns in the file.")
        return
    
    # Extract the specified column's values and label
    column_data = data.iloc[:, column_index]
    column_frame = data.iloc[:, 0]
    column_label = data.columns[column_index]
    
    # Filter out values above the specified max_value
    filtered_data = column_data[column_frame >= 7500]
    filtered_data = filtered_data[filtered_data <= max_value]
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_data.dropna(), bins=bins, edgecolor='black')
    plt.title(f'Histogram of {column_label} (Values ≤ {max_value})')
    plt.xlabel(column_label)
    plt.ylabel('Occurences')
    #plt.yscale('log')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'histogram_{column_label}_'  + str(bins) +'_bins.pdf')
    plt.show()

# Example usage:
file_path = "/Users/thomasborsoni/Desktop/Post-doc/Projet fourmis/Programmes/Data management/file.csv"
plot_histogram_from_csv(file_path, column_index=4, bins=50, max_value=50)


#%%

def plot_histogram_velocities(file_path, bins=1000, max_value=50):
    """
    Reads a CSV file, extracts the specified column (default is the 5th column),
    filters out values above the specified max_value, and plots a histogram of the remaining values.
    
    Parameters:
    - file_path (str): Path to the .csv file.
    - column_index (int): Index of the column to plot (0-based, so 4 is the 5th column).
    - bins (int): Number of bins for the histogram (default is 50).
    - max_value (int or float): Maximum value threshold; values above this will be excluded.
    """
    # Load the CSV file into a DataFrame

    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Check if the file has enough columns
    if data.shape[1] <= column_index:
        print("Error: The specified column index exceeds the number of columns in the file.")
        return
    
    # Extract the specified column's values and label
    positions_x = data.iloc[:, 2]
    positions_y = data.iloc[:, 3]
    column_frame = data.iloc[:, 0]
    column_label = data.columns[4]
    
    # Filter out values before 5 minutes
    filtered_x = positions_x[column_frame >= 7500]
    filtered_y = positions_y[column_frame >= 7500]
    
    
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_data.dropna(), bins=bins, edgecolor='black')
    plt.title(f'Histogram of {column_label} (Values ≤ {max_value})')
    plt.xlabel(column_label)
    plt.ylabel('Occurences')
    #plt.yscale('log')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'histogram_{column_label}_'  + str(bins) +'_bins.pdf')
    plt.show()

# Example usage:
file_path = "/Users/thomasborsoni/Desktop/Post-doc/Projet fourmis/Programmes/Data management/file.csv"
plot_histogram_from_csv(file_path, bins=50, max_value=50)

#%%

data = pd.read_csv(file_path)

# Make sure your data is sorted by time for each agent
data = data.sort_values(by=[data.columns[1], data.columns[0]])

# Calculate the differences in position and time
data['dx'] = data.groupby(data.columns[1])[data.columns[2]].diff()
data['dy'] = data.groupby(data.columns[1])[data.columns[3]].diff()
data['dt'] = data.groupby(data.columns[1])[data.columns[0]].diff()

# Calculate the velocity components (vx and vy)
data['vx'] = data['dx'] / data['dt']
data['vy'] = data['dy'] / data['dt']

# Calculate the total speed (magnitude of the velocity vector)
data['speed'] = np.sqrt(data['vx']**2 + data['vy']**2)


# Print the resulting DataFrame with velocity information
plt.figure(figsize=(10, 6))
plt.hist(np.sqrt(data['vx']**2 + data['vy']**2).dropna(), bins=500, edgecolor='black')
#plt.yscale('log')
plt.grid(axis='y', alpha=0.75)
plt.show()


#%%

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

#%%


def analyze_idle_time(file_path, id_column_index=1, status_column_index=4, threshold=1, bins=100):
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
    #idle_counts =  idle_counts[ idle_counts > 100]    
   # idle_counts =  idle_counts[ idle_counts < 1000]

    
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
    
  #  X = np.linspace(100,999,1000)
  #  plt.plot(X, 200 * np.exp(-X / 200.67))
    
    plt.show()


file_path = "/Users/thomasborsoni/Desktop/Post-doc/Projet fourmis/Programmes/Data management/file.csv"
# Exemple d'utilisation :
analyze_idle_time(file_path)

#%%


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

