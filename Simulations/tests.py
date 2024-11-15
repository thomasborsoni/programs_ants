import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt  
from utils import (is_in_visual_field, 
                   is_in_contact, 
                   blocks_target_from_ref, 
                   get_others_in_visual_field, 
                   resolve_collisions,
                   resolve_collisions_with_inbounds,
                   waiting_step,
                   update_orientation,
                   out_of_bridge_to_other_side)

# %matplotlib qt

#%% Test of the function out_of_bridge_to_other_side

# Set the number of agents
num_agents = 50

# Generate random x and y coordinates within the specified bounds
x_positions = rd.rand(num_agents) * 140 - 20
y_positions = rd.rand(num_agents) * 20

pos_vect = np.empty((num_agents,2))
pos_vect[:,0] = x_positions
pos_vect[:,1] = y_positions

orientations = rd.rand(num_agents) * 2 * np.pi

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))  # Match aspect ratio to grid
ax.set_xlim(-20, 120)
ax.set_ylim(0, 20)

new_p = out_of_bridge_to_other_side(pos_vect, 100)

# Add circles with fixed radius of 1 in data coordinates
radius = 1
for k in range(num_agents):
    
    x, y = x_positions[k] , y_positions[k]
    x2, y2 = new_p[k,0] , new_p[k,1]
    
    if x == x2 and y == y2 : 
        circle = plt.Circle((x, y), radius=radius, color='blue', alpha=0.6)
        ax.add_patch(circle)
        
        plt.arrow(x, y, 
                  3*np.cos(orientations[k]), 3*np.sin(orientations[k]), 
                  head_width=0.1 * 3, head_length=0.2 * 3, fc='grey', ec='grey')
        
    else :
        circle = plt.Circle((x, y), radius=radius, color='orange', alpha=0.6)
        ax.add_patch(circle)
        
        plt.arrow(x, y, 
                  3*np.cos(orientations[k]), 3*np.sin(orientations[k]), 
                  head_width=0.1 * 3, head_length=0.2 * 3, fc='orange', ec='orange')
        
        circle = plt.Circle((x2, y2), radius=radius, color='green', alpha=0.6)
        ax.add_patch(circle)
        plt.arrow(x2, y2, 
                  3*np.cos(orientations[k]), 3*np.sin(orientations[k]), 
                  head_width=0.1 * 3, head_length=0.2 * 3, fc='green', ec='green')
        

# Ensure equal scaling so circles are not distorted
ax.set_aspect('equal')
# Add labels and grid for better readability
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Positions')
plt.legend()
plt.grid(True)
plt.show()





#%% Test of the function waiting_step


# Set the number of agents
num_agents = 50

# Generate random x and y coordinates within the specified bounds
x_positions = rd.rand(num_agents) * 100
y_positions = rd.rand(num_agents) * 20

pos_vect = np.empty((num_agents,2))
pos_vect[:,0] = x_positions
pos_vect[:,1] = y_positions

#%%
frames_since_stop = np.zeros(num_agents).astype(int) + 11
stop_frames_left = np.zeros(num_agents).astype(int)

waiting_step(pos_vect, frames_since_stop, stop_frames_left, frames_since_stop_min = 10, 
                   lbda = 1., distance_collision = 2.1)
print(stop_frames_left)
print(frames_since_stop)


# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))  # Match aspect ratio to grid
ax.set_xlim(0, 100)
ax.set_ylim(0, 20)

# Add circles with fixed radius of 1 in data coordinates
radius = 1

for k in range(num_agents):    
    
    if stop_frames_left[k] > 0:
        circle = plt.Circle((pos_vect[k,0], pos_vect[k,1]), radius=radius, color='orange', alpha=0.6)
        ax.add_patch(circle)
        
    else:
        circle = plt.Circle((pos_vect[k,0], pos_vect[k,1]), radius=radius, color='blue', alpha=0.6)
        ax.add_patch(circle)
    
    
        

# Ensure equal scaling so circles are not distorted
ax.set_aspect('equal')
# Add labels and grid for better readability
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Positions')
plt.legend()
plt.grid(True)
plt.show()








#%% Test of the function "resolve_collisions"

# Set the number of agents
num_agents = 50

# Generate random x and y coordinates within the specified bounds
x_positions = rd.rand(num_agents) * 100
y_positions = rd.rand(num_agents) * 20

pos_vect = np.empty((num_agents,2))
pos_vect[:,0] = x_positions
pos_vect[:,1] = y_positions


# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))  # Match aspect ratio to grid
ax.set_xlim(0, 100)
ax.set_ylim(0, 20)

# Add circles with fixed radius of 1 in data coordinates
radius = 1
for x, y in zip(x_positions, y_positions):
    circle = plt.Circle((x, y), radius=radius, color='blue', alpha=0.6)
    ax.add_patch(circle)

new_pos_vect = resolve_collisions(pos_vect, radius)
other_pos_vect = resolve_collisions_with_inbounds(pos_vect, radius)

for k in range(num_agents):
    
    x,y = new_pos_vect[k,0], new_pos_vect[k,1]
    x2,y2 = other_pos_vect[k,0], other_pos_vect[k,1]
    
    if x2 == pos_vect[k,0] and y2 == pos_vect[k,1]:
        
        a=1
    
    else:
    
        circle = plt.Circle((x, y), radius=radius, color='orange', alpha=0.6)
        ax.add_patch(circle)
    
    if x!=x2 or y!=y2:
        
        circle = plt.Circle((x2, y2), radius=radius, color='red', alpha=0.6)
        ax.add_patch(circle)
        

# Ensure equal scaling so circles are not distorted
ax.set_aspect('equal')
# Add labels and grid for better readability
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Positions')
plt.legend()
plt.grid(True)
plt.show()







#%% Test of the function "get_others_in_visual_field"

# Set the number of agents
num_agents = 100

# Generate random x and y coordinates within the specified bounds
x_positions = rd.rand(num_agents) * 100
y_positions = rd.rand(num_agents) * 20

pos_vect = np.empty((num_agents,2))
pos_vect[:,0] = x_positions
pos_vect[:,1] = y_positions

orientation_ref = rd.rand() * 2 * np.pi



# Set up the plot
fig, ax = plt.subplots(figsize=(10, 2))  # Match aspect ratio to grid
ax.set_xlim(0, 100)
ax.set_ylim(0, 20)

# Add circles with fixed radius of 1 in data coordinates
radius = 1
for x, y in zip(x_positions, y_positions):
    circle = plt.Circle((x, y), radius=radius, color='blue', alpha=0.6)
    ax.add_patch(circle)



# Re-do the plot with ref = 0 and check for all others
circle = plt.Circle((x_positions[0], y_positions[0]), radius=radius, color='black')
ax.add_patch(circle)

# Draw the orientation
plt.arrow(x_positions[0], y_positions[0], 3*np.cos(orientation_ref), 3*np.sin(orientation_ref), head_width=0.1 * 3, head_length=0.2 * 3, fc='green', ec='green')
plt.arrow(x_positions[0], y_positions[0], 3*np.cos(orientation_ref-np.pi/4), 3*np.sin(orientation_ref-np.pi/4), head_width=0.1 * 3, head_length=0.2 * 3, fc='grey', ec='grey')
plt.arrow(x_positions[0], y_positions[0], 3*np.cos(orientation_ref+np.pi/4), 3*np.sin(orientation_ref+np.pi/4), head_width=0.1 * 3, head_length=0.2 * 3, fc='grey', ec='grey')


pos_ref = np.array([x_positions[0], y_positions[0]])

for n in range(2,num_agents):
    
    pos_target = np.array([x_positions[n], y_positions[n]])
    
    if is_in_visual_field(pos_ref, pos_target, orientation_ref, angle_vision=np.pi/2, distance_max=30):
        
        
        circle = plt.Circle((x_positions[n], y_positions[n]), radius=radius, color='red', alpha = .7)
        ax.add_patch(circle)

indices = get_others_in_visual_field(0, orientation_ref, pos_vect, angle_vision = np.pi/2, distance_max = 30, radius = 1)


pos_ref = np.array([x_positions[0], y_positions[0]])

for n in indices:
    
    circle = plt.Circle((x_positions[n], y_positions[n]), radius=radius, color='orange', alpha = .7)
    ax.add_patch(circle)


# Ensure equal scaling so circles are not distorted
ax.set_aspect('equal')
# Add labels and grid for better readability
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Positions')
plt.legend()
plt.grid(True)
plt.show()




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

