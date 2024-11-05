import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def distance(x1, y1, x2, y2):
    """Calcule la distance euclidienne entre deux points (x1, y1) et (x2, y2)."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

@njit
def norme(vect):
    
    return np.sqrt(vect[0] ** 2 + vect[1] ** 2)

@njit
def angle_to(x1, y1, x2, y2, orientation):
    """Calcule l'angle relatif entre deux fourmis par rapport à l'orientation de la fourmi."""
    angle = np.arctan2(y2 - y1, x2 - x1)
    return (angle - orientation + np.pi) % (2 * np.pi) - np.pi

@njit
def vecteur(p1x, p1y, p2x, p2y):
    """Calcule le vecteur entre deux points."""
    return p2x - p1x, p2y - p1y

@njit
def produit_scalaire(v1x, v1y, v2x, v2y):
    """Calcule le produit scalaire entre deux vecteurs."""
    return v1x * v2x + v1y * v2y

@njit
def point_projete(p1x, p1y, p2x, p2y, px, py):
    """Vérifie si un point projeté se trouve sur le segment défini par deux points."""
    vx, vy = vecteur(p1x, p1y, p2x, p2y)
    vpx, vpy = vecteur(p1x, p1y, px, py)
    proj = produit_scalaire(vx, vy, vpx, vpy) / produit_scalaire(vx, vy, vx, vy)
    return 0 <= proj <= 1


@njit
def est_dans_champ_de_vision(x1, y1, orientation, x2, y2, distance_max, angle_vision):
    """Vérifie si une autre fourmi est dans le champ de vision."""
    if distance(x1, y1, x2, y2) > distance_max:
        return False

    angle_rel = angle_to(x1, y1, x2, y2, orientation)
    # L'angle est ramené entre -pi et pi
    angle_rel = (angle_rel + np.pi) % (2 * np.pi) - np.pi

    return (abs(angle_rel) < (angle_vision / 2))

@njit
def angle_rel12(x1, y1, rect2):
    
    angle_rel_min = np.pi
    angle_rel_max = - np.pi
    
    for position2 in rect2:
        
        angle_rel = angle_to(x1, y1, position2[0], position2[1], 0)
        # L'angle est ramené entre -pi et pi
        angle_rel = (angle_rel + np.pi) % (2 * np.pi) - np.pi
        
        angle_rel_min = min(angle_rel_min, angle_rel)
        angle_rel_max = max(angle_rel_max, angle_rel)

    return (angle_rel_min + np.pi) % (2 * np.pi) - np.pi, (angle_rel_max + np.pi) % (2 * np.pi) - np.pi


@njit
def est_occluse(position_i, position_j, position_k, orientation_j, orientation_k, longueur, largeur):
    
    vect_ij = position_j - position_i
    vect_ik = position_k - position_i
    
    if norme(vect_ik) > norme(vect_ij) :
        
        return False
    
    else :
        
        rect_k = get_rectangle(position_k[0], position_k[1], orientation_k, longueur, largeur)
        
        angle_ik_min, angle_ik_max = angle_rel12(position_i[0], position_i[1], rect_k)
    
        # ici on fait une approx pour simplifier et prend la fourmi j comme un point au lieu du rectangle
        
        angle_ij = angle_to(position_i[0], position_i[1], position_j[0], position_j[1], 0)
        
        if angle_ik_min <= angle_ik_max :
            
            print('a')
            
            return (angle_ik_min <= angle_ij <= angle_ik_max)
        
        else :
            print('p')
            return (angle_ik_min <= angle_ij <= angle_ik_max + 2 * np.pi)
        



@njit
def get_rectangle(x, y, orientation, longueur, largeur):
    """Retourne les 4 coins du rectangle représentant une fourmi."""
    dx = np.cos(orientation) * longueur / 2
    dy = np.sin(orientation) * longueur / 2
    dx_perp = np.sin(orientation) * largeur / 2
    dy_perp = -np.cos(orientation) * largeur / 2

    # Calcul des coins du rectangle (4 coins autour du centre de la fourmi)
    return np.array([
        [x - dx - dx_perp, y - dy - dy_perp],  # Coin arrière gauche
        [x - dx + dx_perp, y - dy + dy_perp],  # Coin arrière droit
        [x + dx + dx_perp, y + dy + dy_perp],  # Coin avant droit
        [x + dx - dx_perp, y + dy - dy_perp]   # Coin avant gauche
    ])




#@njit
def detecter_fourmis_dans_champ(fourmis_positions, orientations, longueur, largeur, distance_max, angle_vision):
    """Détecte les fourmis dans le champ de vision de chaque fourmi en tenant compte des occlusions."""
    num_fourmis = len(fourmis_positions)
    resultats = np.empty((num_fourmis, num_fourmis), dtype=np.bool_)

    for i in range(num_fourmis):
        x1, y1 = fourmis_positions[i]
        orientation1 = orientations[i]

        for j in range(num_fourmis):
            if i == j:
                resultats[i, j] = False
                continue

            x2, y2 = fourmis_positions[j]
            orientation2 = orientations[j]

            if est_dans_champ_de_vision(x1, y1, orientation1, x2, y2, distance_max, angle_vision):
                # Vérification des occlusions
                if not est_occluse(i, j, longueur, largeur, fourmis_positions, orientations):
                    resultats[i, j] = True
                else:
                    resultats[i, j] = False
            else:
                resultats[i, j] = False

    return resultats

# Exemple d'utilisation
#fourmis_positions = np.array([[0, 0], [2, 1], [4, -1]])  # Positions des fourmis
#orientations = np.array([0, np.pi / 4, np.pi / 2])  # Orientations des fourmis en radians
#longueurs = np.array([1, 1, 1])  # Longueur des fourmis
#largeurs = np.array([0.5, 0.5, 0.5])  # Largeur des fourmis

Nfourmis = 10
fourmis_positions = np.random.rand(Nfourmis,2)  # Positions des fourmis
orientations = 2 * np.pi * np.random.rand(Nfourmis)  # Orientations des fourmis en radians
longueur = .1  # Longueur des fourmis
largeur = .25 * longueur  # Largeur des fourmis

distance_max = 5
angle_vision = np.pi / 2  # 90 degres  

# Appel de la fonction
#resultats = detecter_fourmis_dans_champ(fourmis_positions, orientations, longueur, largeur, distance_max, angle_vision)


#plt.figure()
#plt.scatter(fourmis_positions[:,0],fourmis_positions[:,1])
#plt.scatter(fourmis_positions[:,0]+ .02*np.cos(orientations),fourmis_positions[:,1] + .02*np.sin(orientations))
#%%
plt.figure()
plt.scatter(fourmis_positions[:,0],fourmis_positions[:,1], color='blue')
plt.scatter(fourmis_positions[0,0],fourmis_positions[0,1], color='green')
plt.scatter(fourmis_positions[0,0]+ .02*np.cos(orientations[0]),fourmis_positions[0,1] + .02*np.sin(orientations[0]), color='orange')
for i in range(1):
    
    x1,y1 = fourmis_positions[i]
    orientation = orientations[i]

    for j in range(Nfourmis):
        
        if j != i :#& i==0 :
            
            x2,y2 = fourmis_positions[j]
            
            rect2 = get_rectangle(x2, y2, orientations[j], longueur, largeur)
        
            amin, amax = angle_rel12(x1, y1, rect2)
            
            distance12 = distance(x1, y1, x2, y2)
            
            plt.scatter(rect2[:,0], rect2[:,1], color='purple')
            
           # plt.scatter(fourmis_positions[i,0]+ distance12*np.cos(amin),fourmis_positions[i,1] + distance12*np.sin(amin), color='red')
          #  plt.scatter(fourmis_positions[i,0]+ distance12*np.cos(amax),fourmis_positions[i,1] + distance12*np.sin(amax), color='red')
            
            for k in range(Nfourmis):
                
                if k != j & k != i :

                
                    if est_occluse(fourmis_positions[i], fourmis_positions[j], fourmis_positions[k], orientations[j], orientations[k], longueur, largeur):
                
                        plt.scatter(fourmis_positions[j,0],fourmis_positions[j,1], color='red')
                        plt.scatter(fourmis_positions[k,0],fourmis_positions[k,1], color='black')

plt.show()







