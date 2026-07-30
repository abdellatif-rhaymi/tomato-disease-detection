# convert_keras_to_tflite.py
import tensorflow as tf
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"DOCKER: Utilisation de TensorFlow version: {tf.__version__}")
logging.info(f"DOCKER: Utilisation de NumPy version: {np.__version__}")

# Chemins à l'intérieur du conteneur Docker
keras_model_path_in_container = '/app/finetuned_best_model.h5'
output_dir_in_container = '/app/output' # Dossier où sauvegarder le TFLite

# Noms des fichiers de sortie
tflite_filename_quant = os.path.join(output_dir_in_container, 'tomato_disease_model_tf214_quant.tflite')
tflite_filename_noquant = os.path.join(output_dir_in_container, 'tomato_disease_model_tf214_NOQUANT.tflite')

# Crée le dossier de sortie s'il n'existe pas DANS le conteneur
os.makedirs(output_dir_in_container, exist_ok=True)

# Vérification et Chargement
logging.info(f"DOCKER: Vérification/Chargement du modèle Keras : {keras_model_path_in_container}")
if not os.path.exists(keras_model_path_in_container):
    logging.error(f"ERREUR DOCKER : Fichier modèle '{keras_model_path_in_container}' non trouvé dans le conteneur.")
    exit("Fichier Keras non trouvé dans /app.")
try:
    model = tf.keras.models.load_model(keras_model_path_in_container)
    logging.info("DOCKER: Modèle Keras chargé.")
except Exception as e:
    logging.error(f"DOCKER: Erreur chargement : {e}")
    exit("Erreur chargement Keras.")

# Conversion AVEC Quantification
logging.info("DOCKER: Conversion TFLite AVEC quantification...")
tflite_quant_model = None
try:
    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter_quant.convert()
    logging.info("DOCKER: Conversion AVEC quantification réussie.")
    with open(tflite_filename_quant, 'wb') as f: f.write(tflite_quant_model)
    logging.info(f"DOCKER: Modèle TFLite quantifié sauvegardé dans : {tflite_filename_quant}")
except Exception as e:
    logging.error(f"DOCKER: Erreur conversion AVEC quantification : {e}")

# Conversion SANS Quantification (si la première a échoué)
if tflite_quant_model is None:
    logging.info("DOCKER: Tentative de conversion SANS quantification...")
    try:
        converter_noquant = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model_noquant = converter_noquant.convert()
        logging.info("DOCKER: Conversion SANS quantification réussie.")
        with open(tflite_filename_noquant, 'wb') as f: f.write(tflite_model_noquant)
        logging.info(f"DOCKER: Modèle TFLite NON quantifié sauvegardé dans : {tflite_filename_noquant}")
    except Exception as e2:
        logging.error(f"DOCKER: Erreur conversion SANS quantification aussi : {e2}")

logging.info("DOCKER: Script de conversion terminé.")