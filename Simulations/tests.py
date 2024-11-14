import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt  # %matplotlib qt
from utils import is_in_visual_field, is_in_contact, blocks_target_from_ref, get_others_in_visual_field
#import time

#%% Test the function "get_others_in_visual_field"

# Set the number of agents
num_agents = 150

# Generate random x and y coordinates within the specified bounds
x_positions = rd.rand(num_agents) * 100
y_positions = rd.rand(num_agents) * 20

pos_vect = np.empty((num_agents,2))
pos_vect[:,0] = x_positions
pos_vect[:,1] = y_positions

orientation_ref = rd.rand() * 2 * np.pi


# Plot the agents in the specified rectangle
plt.figure(figsize=(10, 4))
plt.scatter(x_positions, y_positions, color='blue')
plt.xlim(0, 100)
plt.ylim(0, 20)

# Re-do the plot with ref = 0 and check for all others
plt.scatter(x_positions[0], y_positions[0], color='black')
# Draw the orientation
plt.arrow(x_positions[0], y_positions[0], 3*np.cos(orientation_ref), 3*np.sin(orientation_ref), head_width=0.1 * 3, head_length=0.2 * 3, fc='green', ec='green')
plt.arrow(x_positions[0], y_positions[0], 3*np.cos(orientation_ref-np.pi/4), 3*np.sin(orientation_ref-np.pi/4), head_width=0.1 * 3, head_length=0.2 * 3, fc='grey', ec='grey')
plt.arrow(x_positions[0], y_positions[0], 3*np.cos(orientation_ref+np.pi/4), 3*np.sin(orientation_ref+np.pi/4), head_width=0.1 * 3, head_length=0.2 * 3, fc='grey', ec='grey')


pos_ref = np.array([x_positions[0], y_positions[0]])

for n in range(2,num_agents):
    
    pos_target = np.array([x_positions[n], y_positions[n]])
    
    if is_in_visual_field(pos_ref, pos_target, orientation_ref, angle_vision=np.pi/2, distance_max=30):
        
        plt.scatter(x_positions[n], y_positions[n], color='red')

indices = get_others_in_visual_field(0, orientation_ref, pos_vect, angle_vision = np.pi/2, distance_max = 30, radius = 1)


pos_ref = np.array([x_positions[0], y_positions[0]])

for n in indices:
    
    plt.scatter(x_positions[n], y_positions[n], color='orange')

# Add labels and grid for better readability
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Positions')
plt.legend()
plt.grid(True)
plt.show()


#%% Test of the function "is_in_visual_field"

# Set the number of agents
num_agents = 100

# Generate random x and y coordinates within the specified bounds
x_positions = rd.rand(num_agents) * 100
y_positions = rd.rand(num_agents) * 20

orientation_ref = rd.rand() * 2 * np.pi

# Plot the agents in the specified rectangle
plt.figure(figsize=(10, 4))
plt.scatter(x_positions, y_positions, color='blue')
plt.xlim(0, 100)
plt.ylim(0, 20)

# Re-do the plot with ref = 0 and check for all others
plt.scatter(x_positions[0], y_positions[0], color='black')
# Draw the orientation
plt.arrow(x_positions[0], y_positions[0], 3*np.cos(orientation_ref), 3*np.sin(orientation_ref), head_width=0.1 * 3, head_length=0.2 * 3, fc='green', ec='green')
plt.arrow(x_positions[0], y_positions[0], 3*np.cos(orientation_ref-np.pi/4), 3*np.sin(orientation_ref-np.pi/4), head_width=0.1 * 3, head_length=0.2 * 3, fc='grey', ec='grey')
plt.arrow(x_positions[0], y_positions[0], 3*np.cos(orientation_ref+np.pi/4), 3*np.sin(orientation_ref+np.pi/4), head_width=0.1 * 3, head_length=0.2 * 3, fc='grey', ec='grey')

pos_ref = np.array([x_positions[0], y_positions[0]])

for n in range(2,num_agents):
    
    pos_target = np.array([x_positions[n], y_positions[n]])
    
    if is_in_visual_field(pos_ref, pos_target, orientation_ref, angle_vision=np.pi/2, distance_max=30):
        
        plt.scatter(x_positions[n], y_positions[n], color='red')

# Add labels and grid for better readability
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Positions')
plt.legend()
plt.grid(True)
plt.show()

#%% Test of the function "is_in_contact"

# Set the number of agents
num_agents = 1000

# Generate random x and y coordinates within the specified bounds
x_positions = rd.rand(num_agents) * 100
y_positions = rd.rand(num_agents) * 20

orientation_ref = rd.rand() * 2 * np.pi

# Plot the agents in the specified rectangle
plt.figure(figsize=(10, 4))
plt.scatter(x_positions, y_positions, color='blue')
plt.xlim(0, 100)
plt.ylim(0, 20)

# Re-do the plot with ref = 0 and check for all others
plt.scatter(x_positions[0], y_positions[0], color='black')
# Draw the orientation
plt.arrow(x_positions[0], y_positions[0], 3*np.cos(orientation_ref), 3*np.sin(orientation_ref), head_width=0.1 * 3, head_length=0.2 * 3, fc='green', ec='green')
plt.arrow(x_positions[0], y_positions[0], 3*np.cos(orientation_ref-np.pi/4), 3*np.sin(orientation_ref-np.pi/4), head_width=0.1 * 3, head_length=0.2 * 3, fc='grey', ec='grey')
plt.arrow(x_positions[0], y_positions[0], 3*np.cos(orientation_ref+np.pi/4), 3*np.sin(orientation_ref+np.pi/4), head_width=0.1 * 3, head_length=0.2 * 3, fc='grey', ec='grey')

pos_ref = np.array([x_positions[0], y_positions[0]])

for n in range(2,num_agents):
    
    pos_target = np.array([x_positions[n], y_positions[n]])
    
    if is_in_contact(pos_ref, pos_target, orientation_ref, angle_vision=np.pi/2, distance_max=2):
        
        plt.scatter(x_positions[n], y_positions[n], color='red')

# Add labels and grid for better readability
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Positions')
plt.legend()
plt.grid(True)
plt.show()




#%% Test of the function "blocks_target_from_ref"

# Set the number of agents
num_agents = 100

# Generate random x and y coordinates within the specified bounds
x_positions = rd.rand(num_agents) * 100
y_positions = rd.rand(num_agents) * 20

# Plot the agents in the specified rectangle
plt.figure(figsize=(10, 4))
plt.scatter(x_positions, y_positions, color='blue', label='Agents')
plt.xlim(0, 100)
plt.ylim(0, 20)

# Re-do the plot with ref = 0, target = 1 and check for all others
plt.scatter(x_positions[0], y_positions[0], color='black')
plt.scatter(x_positions[1], y_positions[1], color='green')

pos_ref = np.array([x_positions[0], y_positions[0]])
pos_target = np.array([x_positions[1], y_positions[1]])

for n in range(2,num_agents):
    pos_inter = np.array([x_positions[n], y_positions[n]])
    if blocks_target_from_ref(pos_ref, pos_inter, pos_target):
        plt.scatter(x_positions[n], y_positions[n], color='red')

# Add labels and grid for better readability
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Positions')
plt.legend()
plt.grid(True)
plt.show()

