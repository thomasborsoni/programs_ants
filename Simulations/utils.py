import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from numba import njit
import time

# On commence en supposant les fourmis etre des cercles de rayon radius

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
def is_in_contact(pos_ref, pos_target, orientation_ref, angle_vision = np.pi/2, distance_max = 2):
    
    '''
    Vérifie si target et ref sont en contact. Même algo que pour le champ visuel.
    '''

    return is_in_visual_field(pos_ref, pos_target, orientation_ref, angle_vision, distance_max)

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
def get_others_in_visual_field(ID_ref, orientation_ref, pos_vect, angle_vision = np.pi, distance_max = 100, radius = 1):
    
    n = len(pos_vect)
    
    is_in_visual_field_bool = np.zeros(n).astype(bool)
    
    pos_ref = pos_vect[ID_ref]

    for k in range(n):
        
        if k != ID_ref:
            
            pos_target = pos_vect[k]
        
            is_in_visual_field_bool[k] = is_in_visual_field(pos_ref, pos_target, orientation_ref, angle_vision = angle_vision, distance_max = distance_max)

    IDs_visual_field = np.where(is_in_visual_field_bool)[0]

    
    m = len(IDs_visual_field)
    
    is_not_hidden_bool = np.ones(m).astype(bool)
    
    for k in range(m):
        
        pos_target = pos_vect[IDs_visual_field[k]]
        
        for l in IDs_visual_field:
            
            pos_inter = pos_vect[l]
            
            if norm(pos_ref-pos_inter) * norm(pos_target-pos_inter) != 0:
            
                if blocks_target_from_ref(pos_ref, pos_inter, pos_target, radius = radius):
                
                    is_not_hidden_bool[k] = False
                
                    break

    return IDs_visual_field[is_not_hidden_bool]


























