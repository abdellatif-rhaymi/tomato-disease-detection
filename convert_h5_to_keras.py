import tensorflow as tf
import os
import logging

# Configure le logging pour voir ce qui se passe
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemin vers ton meilleur modèle H5 (celui qui existe déjà)
h5_model_path = 'finetuned_best_model.h5'
# Chemin où tu veux sauvegarder le nouveau modèle au format Keras
keras_model_path = 'finetuned_best_model.keras'

# Vérifie si le fichier H5 existe
logging.info(f"Vérification du fichier H5 d'entrée : {h5_model_path}")
if not os.path.exists(h5_model_path):
    logging.error(f"ERREUR : Le fichier H5 '{h5_model_path}' n'existe pas. Assurez-vous que le fine-tuning précédent a bien créé ce fichier.")
    exit()

# Charge le modèle depuis le fichier H5
logging.info(f"Chargement du modèle depuis {h5_model_path}...")
try:
    model = tf.keras.models.load_model(h5_model_path)
    logging.info("Modèle chargé avec succès depuis H5.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle H5 : {e}")
    exit()

# Sauvegarde le modèle chargé au format Keras natif
logging.info(f"Sauvegarde du modèle au format Keras natif : {keras_model_path}...")
try:
    # Keras utilise l'extension .keras pour déterminer le format de sauvegarde natif
    model.save(keras_model_path)
    logging.info(f"Modèle sauvegardé avec succès au format Keras dans {keras_model_path}")
except Exception as e:
    logging.error(f"Erreur lors de la sauvegarde au format Keras : {e}")
    exit()

logging.info("Conversion de format H5 vers Keras terminée.")