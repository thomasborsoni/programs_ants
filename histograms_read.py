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
    column_label = data.columns[column_index]
    
    # Filter out values above the specified max_value
    filtered_data = column_data[column_data <= max_value]
    
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
plot_histogram_from_csv('file.csv', column_index=4, bins=50, max_value=50)
