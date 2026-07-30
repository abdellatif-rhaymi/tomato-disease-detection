# Dans convert_to_tflite.py

import tensorflow as tf
import os
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

keras_model_path = 'finetuned_best_model.keras'
# Nouveau nom pour la version avec TF Ops autorisées
tflite_filename = 'tomato_disease_model_FLEX.tflite' # Changement ici

# --- Vérification et Chargement (idem) ---
logging.info(f"Vérification de l'existence du modèle Keras : {keras_model_path}")
if not os.path.exists(keras_model_path):
    logging.error(f"ERREUR : Le fichier modèle '{keras_model_path}' n'a pas été trouvé.")
    exit()
logging.info(f"Chargement du modèle Keras depuis : {keras_model_path}")
try:
    model = tf.keras.models.load_model(keras_model_path)
    logging.info("Modèle Keras chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle Keras : {e}")
    exit()

# --- Conversion en TensorFlow Lite en AUTORISANT les TF Ops ---
logging.info("Début de la conversion en TensorFlow Lite en autorisant les TF Ops...")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # --- DÉSACTIVER l'optimisation pour ce test ---
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # logging.info("Optimisations TFLite désactivées pour ce test.")

    # --- AJOUT : Autoriser les opérations TensorFlow si besoin ---
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, # Priorité aux opérations natives TFLite
      tf.lite.OpsSet.SELECT_TF_OPS   # Autorise le fallback vers TF Ops
    ]
    logging.info("SELECT_TF_OPS activé (fallback autorisé).")
    # --------------------------------------------------------

    # Effectuer la conversion
    tflite_flex_model = converter.convert() # Renommée la variable
    logging.info("Conversion TFLite (avec TF Ops autorisées) réussie.")

except Exception as e:
    logging.error(f"Erreur lors de la conversion TFLite : {e}")
    exit()

# --- Sauvegarde du Modèle TFLite ---
logging.info(f"Sauvegarde du modèle TFLite 'Flex' dans : {tflite_filename}")
try:
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_flex_model)
    logging.info("Modèle TFLite 'Flex' sauvegardé avec succès.")

    keras_size = os.path.getsize(keras_model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(tflite_filename) / (1024 * 1024)
    logging.info(f"Taille du modèle Keras original ({keras_model_path}): {keras_size:.2f} Mo")
    logging.info(f"Taille du modèle TFLite 'Flex' ({tflite_filename}): {tflite_size:.2f} Mo")

except Exception as e:
    logging.error(f"Erreur lors de la sauvegarde du fichier TFLite : {e}")

logging.info("Processus de conversion terminé.")