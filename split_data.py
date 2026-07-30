import splitfolders
import os
import logging

# --- Configuration ---
# Chemin vers le dossier contenant tes 10 dossiers de classes (celui que tu veux diviser)
# IMPORTANT : Assure-toi que ce chemin est correct par rapport à l'endroit où tu exécutes le script.
# Si ton script est DANS le dossier qui contient 'tomato', ce chemin devrait être bon.
input_folder = os.path.join('Tomato', 'train')

# Dossier où les nouveaux ensembles train/val/test seront créés
output_folder = 'tomato_split_final'

# Ratios pour la division (Train, Validation, Test)
# Assure-toi que la somme fait 1.0
# Exemple : 70% pour train, 15% pour val, 15% pour test
split_ratio = (0.7, 0.15, 0.15)

# --- Logique du Script ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"Vérification du dossier d'entrée : {input_folder}")
if not os.path.isdir(input_folder):
    logging.error(f"ERREUR : Le dossier d'entrée '{input_folder}' n'a pas été trouvé ou n'est pas un dossier.")
    logging.error("Assurez-vous que le script est exécuté depuis le bon endroit et que le chemin est correct.")
    exit() # Arrête le script si le dossier n'est pas trouvé

logging.info(f"Création du dossier de sortie (s'il n'existe pas) : {output_folder}")
os.makedirs(output_folder, exist_ok=True)

logging.info(f"Début de la division du dataset '{input_folder}' en Train/Val/Test...")
logging.info(f"Ratio : Train={split_ratio[0]*100}%, Val={split_ratio[1]*100}%, Test={split_ratio[2]*100}%")

try:
    # Utilisation de split-folders pour diviser le dataset
    # Il va créer les sous-dossiers train, val, test dans output_folder
    splitfolders.ratio(input_folder,
                       output=output_folder,
                       seed=42, # Rend la division reproductible
                       ratio=split_ratio,
                       group_prefix=None) # Pas besoin de préfixe pour les groupes

    logging.info("-" * 30)
    logging.info(f"Division terminée avec succès !")
    logging.info(f"Les nouveaux ensembles de données se trouvent dans : '{output_folder}'")
    logging.info(f"  -> Dossier d'entraînement : {os.path.join(output_folder, 'train')}")
    logging.info(f"  -> Dossier de validation : {os.path.join(output_folder, 'val')}")
    logging.info(f"  -> Dossier de test : {os.path.join(output_folder, 'test')}")
    logging.info("-" * 30)
    logging.info("Vous pouvez maintenant utiliser ces chemins dans votre script d'entraînement Keras.")

except Exception as e:
    logging.error(f"Une erreur est survenue lors de la division : {e}")
    logging.exception("Traceback de l'erreur :")