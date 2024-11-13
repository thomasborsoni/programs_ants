import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.ion()

file_path = "/Users/thomasborsoni/Desktop/Post-doc/Projet fourmis/Programmes/Data management/file.csv"
frame_number_init = 100000

# Load the data
data = pd.read_csv(file_path)

# Sort data by time and ID to ensure proper order
data = data.sort_values(by=[data.columns[0], data.columns[1]])

data = data.iloc[frame_number_init:].reset_index(drop=True)

# Get unique agent IDs and assign each a random color
unique_ids = data[data.columns[1]].unique()
colors = {agent_id: np.random.rand(3,) for agent_id in unique_ids}

# Set up the figure and axis
fig, ax = plt.subplots()
scat = ax.scatter([], [], c=[], s=50)  # Empty scatter plot for initialization
ax.set_xlim(data[data.columns[2]].min() - 1, data[data.columns[2]].max() + 1)
ax.set_ylim(data[data.columns[3]].min() - 1, data[data.columns[3]].max() + 1)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Agent positions over time")

# Function to update scatter plot at each frame
def update(frame):
    # Filter data for the current time step
    current_data = data[data[data.columns[0]] == frame]
    
    # Extract positions and colors based on agent ID
    x = current_data[data.columns[2]]
    y = current_data[data.columns[3]]
    color = [colors[agent_id] for agent_id in current_data[data.columns[1]]]
    
    # Update scatter plot data
    scat.set_offsets(np.c_[x, y])
    scat.set_color(color)
    return scat,

# Create the animation
frames = sorted(data[data.columns[0]].unique())  # Unique time steps as frames
ani = FuncAnimation(fig, update, frames=frames, interval=10, blit=True, repeat=False)


# Display the animation
plt.show()
