import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import json # Pour charger les class_indices si tu les as sauvegardés

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemin vers le MEILLEUR modèle Keras sauvegardé (celui issu du fine-tuning)
SAVED_MODEL_PATH = 'finetuned_best_model.keras' # Ou .h5 si tu utilises ce format

# Chemins vers tes données de TEST
BASE_DATA_DIR = 'tomato_split_final' # Le dossier contenant train/val/test
TEST_DIR = os.path.join(BASE_DATA_DIR, 'test')

# Paramètres (doivent correspondre à ceux utilisés pour l'entraînement pour le chargement d'image)
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32 # Peut être différent pour l'évaluation, mais affecte le nombre de steps

# Fichiers de sortie pour les résultats
REPORT_FILE = 'classification_report_final.txt'
CONFUSION_MATRIX_FILE = 'confusion_matrix_final.png'
CLASS_INDICES_FILE = 'class_indices.json' # Si tu as sauvegardé les indices depuis l'entraînement

# --- Fonctions Utilitaires (Optionnel, mais peut aider à organiser) ---

def load_trained_model(model_path):
    """Charge le modèle Keras entraîné."""
    logging.info(f"Chargement du modèle depuis : {model_path}")
    if not os.path.exists(model_path):
        logging.error(f"ERREUR : Fichier modèle '{model_path}' non trouvé.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info("Modèle chargé avec succès.")
        model.summary() # Affiche un résumé pour confirmer
        return model
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle : {e}")
        return None

def create_test_generator(test_data_dir, img_width, img_height, batch_size):
    """Crée le générateur de données pour l'ensemble de test."""
    logging.info(f"Création du générateur de test pour le dossier : {test_data_dir}")
    if not os.path.isdir(test_data_dir):
        logging.error(f"ERREUR: Dossier de test '{test_data_dir}' non trouvé.")
        return None

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical', # Important pour y_true_indices
        shuffle=False  # TRÈS IMPORTANT: Ne pas mélanger pour la matrice de confusion
    )
    return test_generator

# --- Script Principal ---
if __name__ == '__main__':
    # 1. Charger le modèle entraîné
    model = load_trained_model(SAVED_MODEL_PATH)
    if model is None:
        exit("Arrêt du script à cause de l'échec du chargement du modèle.")

    # 2. Créer le générateur de données de test
    test_generator = create_test_generator(TEST_DIR, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE)
    if test_generator is None:
        exit("Arrêt du script à cause de l'échec de la création du générateur de test.")

    # 3. Obtenir les vraies étiquettes et les prédictions
    logging.info("Calcul des prédictions sur l'ensemble de test...")
    y_true_indices = test_generator.classes
    num_test_samples = len(y_true_indices)

    steps_for_prediction = num_test_samples // BATCH_SIZE
    if num_test_samples % BATCH_SIZE != 0:
        steps_for_prediction += 1

    logging.info(f"Nombre d'échantillons de test: {num_test_samples}")
    logging.info(f"Batch size pour prédiction: {BATCH_SIZE}")
    logging.info(f"Nombre de steps pour prédiction: {steps_for_prediction}")

    test_generator.reset() # S'assurer qu'il commence au début
    predictions_proba = model.predict(test_generator, steps=steps_for_prediction, verbose=1)
    y_pred_indices = np.argmax(predictions_proba, axis=1)

    # Ajustement des longueurs (sécurité)
    if len(y_pred_indices) > num_test_samples:
        y_pred_indices = y_pred_indices[:num_test_samples]
    elif len(y_pred_indices) < num_test_samples:
        y_true_indices = y_true_indices[:len(y_pred_indices)]

    # 4. Obtenir les noms des classes
    # Option A: Depuis le générateur (si tu es sûr de l'ordre)
    class_names = list(test_generator.class_indices.keys())
    logging.info(f"Noms des classes (depuis test_generator): {test_generator.class_indices}")

    # Option B: Charger depuis un fichier JSON si tu les as sauvegardés pendant l'entraînement
    # if os.path.exists(CLASS_INDICES_FILE):
    #     with open(CLASS_INDICES_FILE, 'r') as f:
    #         class_indices_map_loaded = json.load(f)
    #     # S'assurer que les noms sont dans l'ordre des indices 0, 1, 2...
    #     class_names = [None] * len(class_indices_map_loaded)
    #     for name, index in class_indices_map_loaded.items():
    #         class_names[index] = name
    #     logging.info(f"Noms des classes (depuis {CLASS_INDICES_FILE}): {class_names}")
    # else:
    #     logging.warning(f"Fichier '{CLASS_INDICES_FILE}' non trouvé. Utilisation des noms du test_generator.")
    #     class_names = list(test_generator.class_indices.keys())


    # 5. Générer et afficher le rapport de classification
    logging.info("\n--- Rapport de Classification ---")
    report = classification_report(y_true_indices, y_pred_indices, target_names=class_names, digits=4)
    print(report)
    with open(REPORT_FILE, 'w') as f:
        f.write("Rapport de Classification pour le Set de Test\n")
        accuracy_sklearn = np.sum(y_true_indices == y_pred_indices) / len(y_true_indices)
        f.write(f"Accuracy globale (calculée): {accuracy_sklearn:.4f}\n")
        f.write(report)
    logging.info(f"Rapport de classification sauvegardé dans '{REPORT_FILE}'")

    # 6. Générer et afficher la matrice de confusion
    logging.info("\n--- Matrice de Confusion ---")
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    print("Matrice de Confusion (brute):\n", cm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion sur le Set de Test')
    plt.ylabel('Vraie Étiquette (True Label)')
    plt.xlabel('Étiquette Prédite (Predicted Label)')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE)
    # plt.show() # Décommente si tu veux un affichage interactif
    logging.info(f"Matrice de confusion sauvegardée dans '{CONFUSION_MATRIX_FILE}'")

    logging.info("\nÉvaluation détaillée terminée.")