import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from numba import njit
from quantile_from_laws import inverse_linear, vectorized_cdf


@njit
def positionsToAngles(position_ant, same_eyesight_positions, other_eyesight_positions):
    """
    Converts positions to sorted angles relative to a reference position.
    
    Parameters:
        position_ant (np.ndarray): The reference position, shape (2,).
        same_eyesight_positions (np.ndarray): Positions in the same eyesight group, shape (N, 2).
        other_eyesight_positions (np.ndarray): Positions in the other eyesight group, shape (M, 2).
    
    Returns:
        tuple: Two sorted arrays of angles (for same and other eyesight positions).
    """
    # Calculate relative vectors
    same_vectors = same_eyesight_positions - position_ant
    other_vectors = other_eyesight_positions - position_ant
    
    # Convert vectors to angles
    same_angles = np.arctan2(same_vectors[:, 1], same_vectors[:, 0])  # Angles for same eyesight
    other_angles = np.arctan2(other_vectors[:, 1], other_vectors[:, 0])  # Angles for other eyesight
    
    # Sort the angles
    sorted_same_angles = np.sort(same_angles)
    sorted_other_angles = np.sort(other_angles)
    
    return sorted_same_angles, sorted_other_angles


@njit
def getAntsInChunks(same_angles, other_angles):
    ref_angle = other_angles[0]
    chunk_angles = other_angles - ref_angle
    ant_angles = same_angles - ref_angle

    num_chunks = len(chunk_angles)
    num_ants = len(ant_angles)

    ants_per_chunk = np.zeros(num_chunks)

    chunk_idx, ant_idx = 0, 0

    while chunk_idx < num_chunks and ant_idx < num_ants:
        while ant_angles[ant_idx] <= chunk_angles[chunk_idx]:
            ants_per_chunk[chunk_idx - 1] += 1
            ant_idx += 1

            if ant_idx >= num_ants:
                break

        chunk_idx += 1

    # Add remaining ants to the last chunk
    while ant_idx < num_ants:
        if ant_angles[ant_idx] > chunk_angles[-1]:
            ants_per_chunk[-1] += 1

        ant_idx += 1

    return ants_per_chunk


@njit
def normalizeAngle(x):
    if x > np.pi:
        return x - 2 * np.pi
    elif x < -np.pi:
        return x + 2 * np.pi
    else:
        return x


@njit
def getChunkCenters(other_angles):
    num_chunks = len(other_angles)
    chunk_centers = np.zeros(num_chunks)

    chunk_centers[-1] = np.pi + (other_angles[-1] + other_angles[0]) / 2
    chunk_centers[:-1] = (other_angles[:-1] + other_angles[1:]) / 2

    return chunk_centers


@njit
def getSizeChunks(vect_other):
    
    numberChunks = len(vect_other)
    SizeChunks = np.zeros(numberChunks)
    
    SizeChunks[-1] = 2 * np.pi + (vect_other[0] - vect_other[-1])
        
    SizeChunks[:-1] = vect_other[1:] - vect_other[:-1]
        
    return SizeChunks


@njit
def computeChunkWeights(vect_others, target_dir, sigma_dir=10.0):
    
    chunk_centers = getChunkCenters(vect_others)
    print(chunk_centers)
    SizeChunks = getSizeChunks(vect_others)
    
    dist_to_dir = np.empty_like(chunk_centers)

    for i in range(len(chunk_centers)):
        dist_to_dir[i] = normalizeAngle(chunk_centers[i] - target_dir)

    dir_weights = np.exp(-sigma_dir * dist_to_dir**2) * SizeChunks
    
    return dir_weights / np.sum(dir_weights)


@njit
def computeChunkWeights_unif(chunk_centers, sigma_dir=1.0):
    
    dist_to_dir = chunk_centers.copy()

    for i in range(len(chunk_centers)):
        dist_to_dir[i] = normalizeAngle(chunk_centers[i])

    dir_weights = np.exp(-sigma_dir * chunk_centers**2)
    
    return dir_weights / np.sum(dir_weights)


@njit
def selectChunk(ants_per_chunk, dir_weights, convex_param):
    ant_weights = (ants_per_chunk + 1) / np.sum(ants_per_chunk)

    # Combine weights using geometric averaging
    combined_weights = ant_weights * (dir_weights / ant_weights) ** convex_param

    cumulative_weights = np.cumsum(combined_weights)
    cumulative_weights /= cumulative_weights[-1]

    rand_val = rd.rand()

    selected_chunk = 0
    while cumulative_weights[selected_chunk] < rand_val:
        selected_chunk += 1

    return selected_chunk




# Test parameters
sigma_dir = 0.5
direction_angle = rd.rand() * 2 * np.pi
convex_param = 0.9

position_ant = np.array([0, 0])
same_eyesight_positions = 2 * rd.rand(10, 2) - 1
other_eyesight_positions = 2 * rd.rand(5, 2) - 1

# Assuming positionsToAngles exists and converts positions to angles
sorted_same_angles, sorted_other_angles = positionsToAngles(
    position_ant, same_eyesight_positions, other_eyesight_positions
)

ants_per_chunk = getAntsInChunks(sorted_same_angles, sorted_other_angles)
chunk_centers = getChunkCenters(sorted_other_angles)
dir_weights = computeChunkWeights(chunk_centers, direction_angle, sigma_dir)
selected_chunk = selectChunk(ants_per_chunk, dir_weights, convex_param)


sorted_same_angles += np.pi
sorted_other_angles += np.pi
#%%

@njit
def get_pdf_interact(or_same_vect, or_other_vect, sigma_same = .1, base_value = .2, lbda_other = .1, num_points = 1000):
    
    # Orientation belongs to [0,2 PI)
    
    pdf = np.zeros(num_points) + base_value
    angles = np.linspace(0,2*np.pi, num_points)
    
    for x in or_same_vect :
        
        for k in range(num_points):
            
            pdf[k] += np.exp(-((angles[k] - x + np.pi)%(2*np.pi)-np.pi)**2 / sigma_same**2)
        
    for y in or_other_vect :
        
        for k in range(num_points):
            
            pdf[k] *=  ((angles[k] - y + np.pi)%(2*np.pi)-np.pi)**2 / (lbda_other + ((angles[k] - y + np.pi)%(2*np.pi)-np.pi)**2)
        
    return pdf


sigma_dir = np.pi/2

pdf_interact = get_pdf_interact(sorted_same_angles, sorted_other_angles, sigma_same=.2)

x_pdf = np.linspace(0, 2, len(pdf_interact))

pdf_interact /= np.mean(pdf_interact)

pdf_dir = np.exp(-((x_pdf*np.pi - direction_angle + np.pi)%(2*np.pi) - np.pi)**2/sigma_dir)


pdf_dir /= np.mean(pdf_dir)


pdf_mixed = pdf_dir * pdf_interact

pdf_mixed /= np.mean(pdf_mixed)

cdf_mixed = np.cumsum(pdf_mixed)

cdf_mixed /= cdf_mixed[-1]

n_quantiles = 10000
quantile_vect = inverse_linear(cdf_mixed, n_quantiles)

# Visualization
plt.figure(figsize=(10, 6))

# Plot the angles for ants in the same eyesight
plt.plot(x_pdf,  pdf_mixed, label = 'pdf mixed', color = 'black')

for k in range(400):
    x = int(rd.rand() * n_quantiles)
    
    plt.scatter(quantile_vect[x]*2, -1., 
                color='brown',alpha=0.2)
    
x = int(rd.rand() * n_quantiles)

plt.scatter(quantile_vect[x]*2, -1., 
            color='brown',alpha=0.2, label='samples')


# Plot the angles for ants in the same eyesight
plt.scatter(sorted_same_angles / np.pi, np.ones_like(sorted_same_angles), 
            color='blue', label="Agents same group", alpha=0.7)

# Plot the angles for ants in the other eyesight
plt.scatter(sorted_other_angles / np.pi, np.zeros_like(sorted_other_angles), 
            color='orange', label="Agents other group", alpha=0.7)

# Plot the direction angle
plt.scatter([direction_angle / np.pi], [-0.25], 
            color='red', label="Desired direction", s=100, marker='X', zorder=5)


plt.plot(x_pdf, pdf_interact, linestyle = '-.', color = 'green', label = 'pdf interaction')

plt.plot(x_pdf, pdf_dir, linestyle = '-.', color = 'red', label = 'pdf desired')




# Add chunk boundaries for visualization (optional)
#for angle in sorted_other_angles:
#    plt.axvline(x=angle / np.pi, color='gray', linestyle='--', alpha=0.5)

#plt.legend(loc='upper left')
plt.legend()
plt.grid(alpha=0.3)

# Adjust x-axis ticks to show multiples of π
x_ticks = np.arange(0, 2, 0.5)  # Adjust range as needed
x_tick_labels = [f"{t}π" if t != 0 else "0" for t in x_ticks]
plt.xticks(x_ticks, x_tick_labels)

# Show the plot
plt.tight_layout()
plt.show()



