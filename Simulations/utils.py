import numpy as np
import numpy.random as rd
from numba import njit

# On commence en supposant les fourmis etre des cercles de rayon radius

#### Fonctions de base

@njit
def norm(vector):
    
    return np.sqrt(np.sum(vector**2))

@njit
def vector_to_angle_mpi_pi(vector):
    
    '''
    Outputs the angle of a 2D vector, in the range (-pi, pi]
    '''
    return np.arctan2(vector[1],vector[0])
    


@njit
def relative_angle_0_2pi(pos_ref, pos_target, orientation_ref = 0.):
    
    '''
    Calcule l'angle relatif entre deux agents par rapport à l'orientation du premier, dans le range [0, 2 pi)
    '''
    
    return (vector_to_angle_mpi_pi(pos_target - pos_ref) - orientation_ref) % (2 * np.pi)


##### Fonctions relatives aux collisions et au champ visuel

@njit
def do_collide(pos_1, pos_2, twice_radius):
    
    return (norm(pos_1 - pos_2) < twice_radius)



@njit
def is_in_visual_field(pos_ref, pos_target, orientation_ref, angle_vision = np.pi, distance_max = 100):
    
    '''
    Vérifie si target est dans le cone de vision de ref. 
    La valeur angle_vision represente l'angle entre le point le plus a gauche et le point le plus a droite.
    En premiere approx on neglige ici le volume de la fourmi
    '''
    
    if norm(pos_target - pos_ref) > distance_max:
        return False

    angle_rel = relative_angle_0_2pi(pos_ref, pos_target, orientation_ref)
    
    # L'angle est ramené entre -pi et pi
    angle_rel = (angle_rel + np.pi) % (2 * np.pi) - np.pi

    return (abs(angle_rel) < (angle_vision / 2))

@njit
def is_in_contact(pos_ref, pos_target, orientation_ref, angle_vision = np.pi/2, distance_contact = 2):
    
    '''
    Vérifie si target et ref sont en contact. Même algo que pour le champ visuel.
    '''

    return is_in_visual_field(pos_ref, pos_target, orientation_ref, angle_vision, distance_contact)

@njit
def blocks_target_from_ref(pos_ref, pos_inter, pos_target, radius = 1):
    
    '''
    Pour deux agents inter et target dans le champ de vision de ref, verifie si inter cache target
    '''
    
    distance_ref_inter = norm(pos_inter - pos_ref)
    
    if norm(pos_target - pos_ref) < distance_ref_inter:
        return False
    
    
    # Angle relatif entre vect ref -> inter et vect ref -> target
    angle_rel = relative_angle_0_2pi(pos_ref, pos_target, relative_angle_0_2pi(pos_ref, pos_inter))
    
    # L'angle est ramené entre -pi et pi
    angle_rel = (angle_rel + np.pi) % (2 * np.pi) - np.pi
    
    # Angle sous lequel inter cache target
    angle_blocked = abs(np.arctan(radius / distance_ref_inter))

    return (abs(angle_rel) < angle_blocked)




@njit
def get_others_in_visual_field(ID_ref, orientation_ref, pos_vect, angle_vision=np.pi, distance_max=100, radius=1):
    
    n = len(pos_vect)
    
    # Initialize the array with integers instead of booleans
    is_in_visual_field_bool = np.zeros(n, dtype=np.int32)
    
    pos_ref = pos_vect[ID_ref]

    for k in range(n):
        if k != ID_ref:
            pos_target = pos_vect[k]
            is_in_visual_field_bool[k] = is_in_visual_field(pos_ref, pos_target, orientation_ref, 
                                                            angle_vision=angle_vision, 
                                                            distance_max=distance_max)

    # Get indices of True values (i.e., 1 values) in the integer array
    IDs_visual_field = np.where(is_in_visual_field_bool == 1)[0]
    
    m = len(IDs_visual_field)
    
    # Initialize the is_not_hidden_bool as an integer array
    is_not_hidden_bool = np.ones(m, dtype=np.int32)
    
    for k in range(m):
        pos_target = pos_vect[IDs_visual_field[k]]
        
        for l in IDs_visual_field:
            pos_inter = pos_vect[l]
            
            # Avoid division by zero by checking that norms are not zero
            if norm(pos_ref - pos_inter) * norm(pos_target - pos_inter) != 0:
                if blocks_target_from_ref(pos_ref, pos_inter, pos_target, radius=radius):
                    is_not_hidden_bool[k] = 0  # Set to 0 (False) if blocked
                    break

    # Return IDs where is_not_hidden_bool is 1 (i.e., True)
    return IDs_visual_field[is_not_hidden_bool == 1]



##### Gestion de la collision

@njit
def resolve_collisions(pos_vect, radius = 1., epsilon = 1e-5):
    
    n = len(pos_vect)
    
    new_pos_vect = pos_vect.copy()
    
    twice_radius = 2 * radius
    
    for k in range(n) :
        
        pos_1 = new_pos_vect[k]

        l = 0
        
        while l < n:
            
            if l != k :
            
                pos_2 = new_pos_vect[l]
                
                dist_12 = norm(pos_2 - pos_1)
                
    
                if dist_12 < twice_radius :
                    
                    
                    if dist_12 < 1e-15:
                       
                        
                        angle_rand = rd.rand() * 2 * np.pi
                        
                        vect_correct = (radius + epsilon) * np.array([np.cos(angle_rand), np.sin(angle_rand)])
                    
                    else :
                        
                        
                        # Si collision, on 
                        
                        vect_correct = (radius + epsilon - .5 * dist_12 ) * (pos_2 - pos_1) / dist_12
                    
                    new_pos_vect[k] = pos_1 - vect_correct
                    new_pos_vect[l] = pos_2 + vect_correct
                    
                    pos_1 = new_pos_vect[k]
                    
                    l = 0
                
                else :
                    
                    l += 1
            
            else :
                
                l += 1

                
    return new_pos_vect


@njit
def inbounds(pos, xlim=100., ylim = 20., radius = 1.):
    
    new_pos = np.empty(2)
    
    new_pos[0] = min(max(pos[0], radius), xlim - radius)
    new_pos[1] = min(max(pos[1], radius), ylim - radius)
    
    return new_pos


@njit
def resolve_collisions_with_inbounds(pos_vect, radius = 1., epsilon = 1e-5):
    
    n = len(pos_vect)
    
    new_pos_vect = np.zeros((n,2))
    
    for k in range(n):
        
        new_pos_vect[k] = inbounds(pos_vect[k])
    
    twice_radius = 2 * radius
    
    for k in range(n) :
        
        pos_1 = new_pos_vect[k]

        l = 0
        
        while l < n:
            
            if l != k :
            
                pos_2 = new_pos_vect[l]
                
                dist_12 = norm(pos_2 - pos_1)
                
    
                if dist_12 < twice_radius :
                    
                    
                    if dist_12 < 1e-15:
                       
                        
                        angle_rand = rd.rand() * 2 * np.pi
                        
                        vect_correct = (radius + epsilon) * np.array([np.cos(angle_rand), np.sin(angle_rand)])
                    
                    else :
                        
                        
                        # Si collision, on 
                        
                        vect_correct = (radius + epsilon - .5 * dist_12 ) * (pos_2 - pos_1) / dist_12
                    
                    new_pos_vect[k] = inbounds(pos_1 - vect_correct)
                    new_pos_vect[l] = inbounds(pos_2 + vect_correct)
                    
                    pos_1 = new_pos_vect[k]
                    
                    l = 0
                
                else :
                    
                    l += 1
            
            else :
                
                l += 1

                
    return new_pos_vect



@njit
def stop_frames(lbda, fps=25):
    
    return int(fps * rd.exponential(lbda)) + 1

@njit
def update_orientation(pos_vect, new_pos_vect, orientations):
    
    new_orientations = orientations.copy()
    
    for k in range(len(orientations)):
        
        diff = new_pos_vect[k] - pos_vect[k]
        length = norm(diff)
        
        if length > 1e-15:
            
            new_orientations[k] = diff / length
            
    return new_orientations


@njit
def waiting_and_restart_step(pos_vect, frames_since_stop, stop_frames_left,
                   frames_since_stop_min = 10, lbda = 1., distance_collision = 2.1):

    # Indices pour lesquels on n'est pas a l'arret et on a le droit de se mettre a l'arret
    indicies_left = np.where((stop_frames_left == 0) & (frames_since_stop >= frames_since_stop_min))[0]
   
    
    restart_indices = np.where(stop_frames_left ==1)[0]
    
    # Pour les indices a l'arret, on decremente le compteur de frames restantes un
    stop_frames_left[stop_frames_left > 0] -= 1
    
    # Pour les indices qui n'ont pas le droit de s'arreter, on incremente le compteur de frames
    # depuis le dernier arret de un
    frames_since_stop[frames_since_stop < frames_since_stop_min] += 1
    
    # On retiendra pour le suite les indices que l'on a deja mis a l'arret
    stopped = np.zeros_like(indicies_left)
    
    # Sur les indices qui pourraient se mettre a l'arret, on boucle pour savoir si on doit le mettre a l'arret
    for k in indicies_left:
        
        pos_k = pos_vect[k]
        
        # on verifie qu'on a pas deja mis k a l'arret
        
        if stopped[k] > 0:
            continue
       
        for l in indicies_left :
        
            if l == k or stopped[l] > 0 :
                continue
        
            if do_collide(pos_k, pos_vect[l], distance_collision)  :
            
                stopping_time = stop_frames(lbda)
                
                # On arrete les deux indices pour le meme temps
                
                stop_frames_left[k] = stopping_time
                stop_frames_left[l] = stopping_time
                
                # On remet a zero le compteur de frames depuis dernier arret
                frames_since_stop[k] = 0
                frames_since_stop[l] = 0
                
                # On s'assure qu'on ne re-visitera pas k et l
                stopped[k] = 1
                stopped[l] = 1
                
                # On ne communique qu'avec un
                
                break
    
    return frames_since_stop, stop_frames_left, restart_indices
    
    
# Gestion de la sortie du pont, en condition periodique

@njit
def out_of_bridge_to_other_side_with_orientation(positions, orientations, length_x):
    
    n = len(positions)
    
    for k in range(n):
        
        if positions[k,0] > 100 :
            
            if np.cos(orientations[k]) >= 0 :
                
                positions[k,0] = positions[k,0] % 100
                
        elif positions[k,0] < 0 :
            
            if np.cos(orientations[k]) <= 0 :
                
                positions[k,0] = positions[k,0] % 100

    return 0


@njit
def out_of_bridge_to_other_side(positions, length_x = 100.):
    
    n = len(positions)
    
    for k in range(n):
        
        if positions[k,0] > 100 :
                
            positions[k,0] = positions[k,0] % 100
                
        elif positions[k,0] < 0 :
                
            positions[k,0] = positions[k,0] % 100

    return 0




@njit
def new_orientations(n):
    
    new_angles = rd.rand(n) * 2 * np.pi
    angles2D = np.empty((n,2))
    angles2D[:,0] = np.cos(new_angles)
    angles2D[:,1] = np.sin(new_angles)
    
    return angles2D

@njit
def bounce(pos_vect, orientations2D, length_y=20., radius = 1.):
    
    n = len(pos_vect)
    
    for k in range(n):
        
        if pos_vect[k,1] > length_y - radius:
            
            orientations2D[k,1] = - abs(orientations2D[k,1])
            
            
        elif pos_vect[k,1] < radius:
            
            orientations2D[k,1] = abs(orientations2D[k,1])
            
            
    return orientations2D

















