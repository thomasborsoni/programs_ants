## Merci  ChatGPT...

import matplotlib.pyplot as plt
import cv2

# Variables globales pour stocker les positions et les IDs
positions = []
current_fourmi_id = None

# Fonction de callback pour enregistrer les clics
def on_click(event):
    global current_fourmi_id
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        # On demande l'ID de la fourmi si ce n'est pas déjà fait
        if current_fourmi_id is None:
            current_fourmi_id = input("Entrer l'ID de la fourmi : ")
        
        print(f"Fourmi ID {current_fourmi_id}: Position ({x}, {y})")
        
        # Demander si le clic est pour les antennes ou la queue
        label = input("Est-ce la position des antennes (A) ou de la queue (Q) ? ").upper()
        if label == 'A':
            label = 'antennes'
        elif label == 'Q':
            label = 'queue'
        else:
            print("Entrée invalide. Position ignorée.")
            return
        
        # Sauvegarder les informations dans la liste
        positions.append({
            "fourmi_id": current_fourmi_id,
            "x": x,
            "y": y,
            "label": label
        })
        
        # Afficher un marqueur à l'endroit du clic
        plt.plot(x, y, 'ro' if label == 'antennes' else 'bo')
        plt.draw()

# Fonction pour afficher une frame et récupérer les clics
def label_frame(frame_path):
    global positions, current_fourmi_id
    positions = []  # Reset positions for each frame
    current_fourmi_id = None  # Reset fourmi ID for each frame

    img = cv2.imread(frame_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Afficher l'image
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    
    # Lier l'événement de clic
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.show()
    
    # Déconnecter l'événement après la fermeture de la fenêtre
    fig.canvas.mpl_disconnect(cid)
    
    return positions

# Exemple d'utilisation sur une frame
frame_path = "frames/frame_0001.png"
positions = label_frame(frame_path)
print("Positions enregistrées:", positions)
