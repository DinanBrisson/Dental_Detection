import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io

# Charger le modèle YOLOv8
model = YOLO("best_yolo_dental.pt.pt")

st.title("Détection de dents")
st.write("Chargez une image pour effectuer une détection.")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_cv2 = np.array(image.convert("RGB"))  # Conversion PIL -> NumPy

    # Afficher l'image originale
    st.subheader("Image Originale")
    st.image(image, caption="Image chargée", use_container_width=True)

    if st.button("Lancer la détection"):
        try:
            # Exécuter la détection
            results = model(image_cv2)
            annotated_image = results[0].plot()

            # Convertir en image PIL
            annotated_pil = Image.fromarray(annotated_image)
            st.subheader("Image avec les dents détectées")
            st.image(annotated_pil, caption="Image avec détection", use_container_width=True)

            # Stocker l'image en mémoire pour le téléchargement
            img_bytes = io.BytesIO()
            annotated_pil.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            # Bouton de téléchargement
            st.download_button(
                label="Télécharger l'image annotée",
                data=img_bytes,
                file_name="detection_result.jpg",
                mime="image/jpeg"
            )
        except Exception as e:
            st.error(f"Erreur lors de la détection : {e}")
