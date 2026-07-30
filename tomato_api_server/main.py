# main.py
import os
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # Adapte si tu n'as pas utilisé MobileNetV2 pour le preprocessing exact
import numpy as np
from PIL import Image
import io

# Initialisation de Flask
app = Flask(__name__)

# Charger le modèle Keras (chemin relatif au script main.py)
# Le modèle sera copié dans l'image Docker plus tard
MODEL_PATH = 'finetuned_best_model.keras' # Ou .h5
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modèle Keras chargé depuis {MODEL_PATH}")
except Exception as e:
    print(f"Erreur lors du chargement du modèle Keras: {e}")
    model = None # Marquer que le modèle n'est pas chargé

# Dimensions d'entrée attendues par le modèle
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Labels des classes DANS LE BON ORDRE
# !!! METS ICI TA LISTE EXACTE DE LABELS !!!
CLASS_LABELS = [
    "Tomato Bacterial spot", "Tomato Early blight", "Tomato Healthy",
    "Tomato Late blight", "Tomato Leaf Mold", "Tomato Mosaic virus",
    "Tomato Septoria leaf spot", "Tomato Spider mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus"
]

def preprocess_image_for_keras(img_bytes):
    """Prétraite les bytes de l'image pour le modèle Keras."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img) # Convertit en tableau NumPy
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        # Utilise la fonction de preprocessing spécifique à ton modèle de base
        # Si tu as utilisé MobileNetV2 et son preprocess_input pendant l'entraînement, utilise-le.
        # Sinon, si tu as juste fait rescale=1./255, fais ça :
        # return img_array_expanded_dims / 255.0
        return preprocess_input(img_array_expanded_dims) # Pour MobileNetV2, normalise entre -1 et 1
    except Exception as e:
        print(f"Erreur de prétraitement de l'image: {e}")
        return None

@app.route('/')
def hello():
    return "API de Détection de Maladies de Tomates Fonctionne !"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modèle non chargé sur le serveur'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier image fourni'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    if file:
        try:
            img_bytes = file.read()
            processed_image = preprocess_image_for_keras(img_bytes)

            if processed_image is None:
                return jsonify({'error': 'Erreur lors du prétraitement de l_image'}), 500

            predictions = model.predict(processed_image)
            predicted_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_index]) # Convertir en float Python standard

            if predicted_index < len(CLASS_LABELS):
                predicted_label = CLASS_LABELS[predicted_index]
                return jsonify({
                    'predicted_label': predicted_label,
                    'confidence': confidence
                })
            else:
                return jsonify({'error': 'Index de prédiction invalide'}), 500

        except Exception as e:
            print(f"Erreur pendant la prédiction : {e}")
            return jsonify({'error': f'Erreur serveur pendant la prédiction: {str(e)}'}), 500

    return jsonify({'error': 'Erreur inconnue'}), 500

if __name__ == '__main__':
    # Le port est souvent fourni par l'environnement Cloud (ex: PORT variable d'env)
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)