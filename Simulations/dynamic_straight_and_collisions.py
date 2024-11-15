import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
from matplotlib.animation import FuncAnimation, FFMpegWriter
from utils import (is_in_visual_field, 
                   is_in_contact, 
                   blocks_target_from_ref, 
                   get_others_in_visual_field, 
                   resolve_collisions,
                   resolve_collisions_with_inbounds,
                   waiting_and_restart_step,
                   update_orientation,
                   out_of_bridge_to_other_side,
                   new_orientations,
                   bounce)

import os

# Set the path to the location of ffmpeg
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

### Constant parameters
radius = 1.0
lbda = 1.
num_agents = 25
frames_since_stop_min = 100
distance_collision = 2.1
angle_vision = np.pi
distance_max = 30
length_x = 100
length_y = 20
avg_vel = 20

Tfinal = 10  # in seconds
fps = 25

############## Initialize

# Generate random x and y coordinates within the specified bounds
x_positions = rd.rand(num_agents) * length_x
y_positions = rd.rand(num_agents) * length_y

pos_vect = np.empty((num_agents, 2))
pos_vect[:, 0] = x_positions
pos_vect[:, 1] = y_positions
    
# Generate random initial orientations
orientations = rd.rand(num_agents) * 2 * np.pi
orientations2D = np.empty((num_agents, 2))
orientations2D[:, 0] = np.cos(orientations)
orientations2D[:, 1] = np.sin(orientations)

# Get time_steps vector
nb_frames = int(np.ceil(fps * Tfinal))
time_steps = np.linspace(0, Tfinal, nb_frames)
dt = time_steps[1] - time_steps[0]  # Calculate dt from time steps

# Agent colors

colors = {agent_id: np.random.rand(3,) for agent_id in np.arange(num_agents)}


# Initialize frames to count

frames_since_stop = np.zeros(num_agents).astype(int) + frames_since_stop_min + 1
stop_frames_left = np.zeros(num_agents).astype(int)

####### Run

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10,6))
ax.set_aspect('equal')
ax.set_xlim(0, length_x)
ax.set_ylim(0, length_y)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Agents")

# Lists to keep track of Circle and Arrow patches for each frame
circles = []
arrows = []

    
# Function to update scatter plot at each frame
def update(frame, pos_vect, orientations2D, frames_since_stop, stop_frames_left):
    # Clear previous circles from the plot
    global circles, arrows
    # Clear previous circles and arrows from the plot
    for circle in circles:
        circle.remove()
    for arrow in arrows:
        arrow.remove()
    circles = []
    arrows = []

    # Update positions based on velocity and orientation execept for the ones that are stopped
    pos_vect[stop_frames_left==0] += dt * avg_vel * orientations2D[stop_frames_left==0]
    
    # Use perioicity in x
    out_of_bridge_to_other_side(pos_vect, length_x)

    # Use bouncing bc in y
    orientations2D = bounce(pos_vect, orientations2D)

    
    
    # Check for possible discussions and updates
    frames_since_stop, stop_frames_left, restart_indices = waiting_and_restart_step(pos_vect, 
                                                                                    frames_since_stop, stop_frames_left,
                                                                                    frames_since_stop_min, lbda, distance_collision)
    
    # Renew orientation for the restarting agents
    orientations2D[restart_indices] = new_orientations(len(restart_indices))
    
    # Resolve possible collisions
    pos_vect = resolve_collisions_with_inbounds(pos_vect, radius)
    
    # Create and add new circles for each agent
    for agent_id in range(num_agents):
        x, y = pos_vect[agent_id]
        color = colors[agent_id]
        circle = Circle((x, y), radius=radius, color=color, alpha=0.6)
        ax.add_patch(circle)
        circles.append(circle)
        
        dx, dy = orientations2D[agent_id] * (radius * 1.5)  # Arrow length scaled by radius
        arrow = FancyArrow(x, y, dx, dy, width=0.1, color=color, alpha=0.8)
        ax.add_patch(arrow)
        arrows.append(arrow)
    
    return circles + arrows

# Run the animation with blit=False
ani = FuncAnimation(fig, update, frames=time_steps,
                    fargs=(pos_vect, orientations2D, frames_since_stop, stop_frames_left,), 
                    interval=40, blit=True, repeat=False)
ani.save("agent_simulation_1.mp4", writer=FFMpegWriter(fps=25))
# Show the animation
plt.show()
