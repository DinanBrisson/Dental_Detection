import cv2
from ultralytics import YOLO

# Charger le modèle YOLOv8 entraîné
model = YOLO("best_yolo_dental.pt")

# Ouvrir la webcam (0 pour la caméra intégrée, 1 ou 2 pour d'autres caméras externes)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

while True:
    # Lire une frame de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture de l'image.")
        break

    # Appliquer la détection
    results = model(frame)

    # Annoter l'image avec les résultats de YOLO
    annotated_frame = results[0].plot()

    # Afficher l'image avec détection en temps réel
    cv2.imshow("Détection en temps réel - YOLOv8", annotated_frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermer la caméra et les fenêtres OpenCV
cap.release()
cv2.destroyAllWindows()
