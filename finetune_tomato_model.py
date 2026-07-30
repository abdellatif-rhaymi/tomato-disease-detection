import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt
import numpy as np
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_dir = 'tomato_split_final'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

base_model_path = 'finetuned_best_model.h5'
finetuned_model_save_path = 'finetuned_best_model.keras' # Changer l'extension
# ...
history_plot_path = 'finetuning_history2.png'

IMG_WIDTH, IMG_HEIGHT = 224, 224
NUM_CLASSES = 10
BATCH_SIZE = 32
FINETUNE_EPOCHS = 8
LOW_LEARNING_RATE = 1e-5 # 0.00001

# --- Chargement du Modèle Pré-entraîné (Phase 1) ---
logging.info(f"Chargement du modèle de base depuis : {base_model_path}")
try:
    model = load_model(base_model_path)
    logging.info("Modèle chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle '{base_model_path}': {e}")
    exit()

# --- Dégel (Unfreeze) des Couches ---
# On va itérer sur les couches du modèle chargé et dégeler celles qui appartiennent à MobileNetV2
# On peut identifier les couches de fin ajoutées par nous pour ne PAS les geler/dégeler inutilement.
# Les couches ajoutées sont typiquement à la toute fin.
layers_to_skip_for_freezing = ['global_average_pooling2d', 'dropout', 'dense'] # Noms de tes couches ajoutées

# Détermine l'index à partir duquel dégeler.
# Trouvons une couche de fin de MobileNetV2 dans la liste, par exemple 'block_16_project_BN' ou 'Conv_1'
# (Prends un nom dans la liste d'erreur qui semble être vers la fin des blocs MobileNetV2)
unfreeze_from_layer_name = 'block_14_expand' # Exemple: on dégèle à partir du bloc 14. Ajuste ce nom !

unfreeze_start_index = -1
logging.info(f"Recherche de l'index de la couche '{unfreeze_from_layer_name}' pour le dégel...")
for i, layer in enumerate(model.layers):
    if layer.name == unfreeze_from_layer_name:
        unfreeze_start_index = i
        break

if unfreeze_start_index == -1:
    logging.warning(f"AVERTISSEMENT : Couche '{unfreeze_from_layer_name}' non trouvée. Dégel de toutes les couches sauf les dernières.")
    # Alternative si on ne trouve pas le nom: dégeler tout sauf les dernières ajoutées.
    # Compte les couches à sauter : len(layers_to_skip_for_freezing)
    num_layers_total = len(model.layers)
    unfreeze_start_index = num_layers_total - len(layers_to_skip_for_freezing) - 30 # Dégèle 30 couches avant les couches ajoutées

logging.info(f"Nombre total de couches dans le modèle : {len(model.layers)}")
logging.info(f"Dégel des couches à partir de l'index : {unfreeze_start_index}")

# Dégeler les couches appropriées
for i, layer in enumerate(model.layers):
    # Ne pas toucher aux couches que nous avons ajoutées à la fin
    if layer.name in layers_to_skip_for_freezing:
        layer.trainable = True # S'assurer qu'elles sont entraînables
        logging.debug(f"Couche {layer.name} (ajoutée) est trainable.")
    # Geler les couches AVANT l'index de dégel
    elif i < unfreeze_start_index:
        layer.trainable = False
        logging.debug(f"Couche {layer.name} gelée.")
    # Dégeler les couches À PARTIR de l'index de dégel
    else:
        layer.trainable = True
        logging.debug(f"Couche {layer.name} dégelée.")


# --- Re-Compilation avec Faible Learning Rate ---
logging.info(f"Re-compilation du modèle avec un learning rate de {LOW_LEARNING_RATE}")
model.compile(optimizer=Adam(learning_rate=LOW_LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

logging.info("Résumé du modèle après dégel partiel:")
model.summary() # Doit montrer plus de paramètres entraînables maintenant

# --- Préparation des Générateurs de Données (Identique à avant) ---
# (Le reste du code pour les générateurs, le checkpoint, model.fit, l'évaluation, etc. reste identique)
# ... (colle le reste du script à partir d'ici) ...

logging.info("Préparation des générateurs de données...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)



finetune_checkpoint = ModelCheckpoint(
    filepath=finetuned_model_save_path, # Utiliser le nouveau nom
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    #save_format='tf' # Optionnel mais peut aider à forcer le nouveau format
)

# --- Fine-Tuning ---
logging.info(f"Début du Fine-Tuning pour {FINETUNE_EPOCHS} époques...")

history_finetune = model.fit(
    train_generator,
    epochs=FINETUNE_EPOCHS,
    validation_data=validation_generator,
    callbacks=[finetune_checkpoint]
)

logging.info("Fine-tuning terminé.")

# --- Visualisation de l'Historique du Fine-Tuning ---
logging.info("Génération des graphiques d'historique du fine-tuning...")
acc = history_finetune.history['accuracy']
val_acc = history_finetune.history['val_accuracy']
loss = history_finetune.history['loss']
val_loss = history_finetune.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Fine-Tuning Accuracy')
plt.xlabel(f'Epochs ({len(acc)} total)')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Fine-Tuning Loss')
plt.xlabel(f'Epochs ({len(acc)} total)')
plt.ylabel('Loss')

plt.suptitle('Fine-Tuning History')
plt.savefig(history_plot_path)
logging.info(f"Graphiques sauvegardés dans '{history_plot_path}'")
logging.info(f"Le meilleur modèle fine-tuné a été sauvegardé dans '{finetuned_model_save_path}'")


# --- Évaluation Finale sur le Set de Test (avec le modèle fine-tuné) ---
logging.info(f"\nÉvaluation du meilleur modèle fine-tuné ({finetuned_model_save_path}) sur le set de test...")

# Charger le MEILLEUR modèle fine-tuné
try:
    best_finetuned_model = tf.keras.models.load_model(finetuned_model_save_path)

    # Préparer le générateur de test
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False)

    # Évaluer
    loss_test, accuracy_test = best_finetuned_model.evaluate(test_generator)

    print("-" * 30)
    print(f"Performance FINALE (après fine-tuning) sur le set de test :")
    print(f"  -> Perte (Loss)    : {loss_test:.4f}")
    print(f"  -> Précision (Acc) : {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print("-" * 30)

except Exception as e:
    logging.error(f"Erreur lors du chargement ou de l'évaluation du modèle fine-tuné : {e}")

logging.info("Processus de fine-tuning terminé.")