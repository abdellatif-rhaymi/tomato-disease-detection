import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # Ou EfficientNetLiteB0, etc.
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Chemins vers tes données (créés par split_folders)
base_dir = 'tomato_split_final' # Le dossier contenant train/val/test
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test') # Gardé pour l'évaluation finale

# Paramètres du modèle et de l'entraînement
IMG_WIDTH, IMG_HEIGHT = 224, 224 # Taille d'image attendue par MobileNetV2
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
NUM_CLASSES = 10 # Important: Doit correspondre à tes 10 dossiers de classes
BATCH_SIZE = 32  # Ajustable (16, 32, 64...) selon ta mémoire GPU
EPOCHS = 30      # Nombre max d'époques (EarlyStopping peut l'arrêter avant)
LEARNING_RATE = 0.0001 # Un learning rate plus bas est souvent bon pour le fine-tuning

# --- Préparation des Données (ImageDataGenerator) ---
print("Préparation des générateurs de données...")

# Augmentation pour l'entraînement (légère au début, car données déjà augmentées ?)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # rotation_range=20, # Commence sans ou avec peu
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # shear_range=0.1,
    # zoom_range=0.1,
    horizontal_flip=True, # Une augmentation souvent sûre
    fill_mode='nearest'
)

# Seulement rescale pour la validation et le test
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Création des générateurs
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical' # Pour multi-classe
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Pas besoin de mélanger la validation
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Pas besoin de mélanger le test
)

# Vérifier les classes trouvées (important pour mapper les sorties plus tard)
print("Classes trouvées par le générateur d'entraînement :")
print(train_generator.class_indices)
# Sauvegarder les indices de classe pour une utilisation future
class_indices = train_generator.class_indices
# Peut-être sauvegarder dans un fichier json si besoin
# import json
# with open('class_indices.json', 'w') as f:
#    json.dump(class_indices, f)


# --- Construction du Modèle (Transfer Learning) ---
print("Construction du modèle...")

# Charger MobileNetV2 pré-entraîné sur ImageNet, sans la couche du dessus
base_model = MobileNetV2(input_shape=IMG_SHAPE,
                           include_top=False, # Très important
                           weights='imagenet')

# Geler les poids du modèle de base (on ne les ré-entraîne pas au début)
base_model.trainable = False

# Ajouter nos propres couches de classification
x = base_model.output
x = GlobalAveragePooling2D()(x) # Réduit la dimensionnalité
x = Dropout(0.5)(x) # Régularisation pour éviter le surapprentissage
outputs = Dense(NUM_CLASSES, activation='softmax')(x) # Couche finale avec 10 neurones

# Créer le modèle final
model = Model(inputs=base_model.input, outputs=outputs)

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Résumé du modèle :")
model.summary()

# --- Callbacks (pour améliorer l'entraînement) ---
# Arrêter l'entraînement si la performance sur la validation ne s'améliore plus
early_stopping = EarlyStopping(monitor='val_loss', # Ou 'val_accuracy'
                               patience=5,         # Nbre d'époques sans amélioration avant d'arrêter
                               verbose=1,
                               restore_best_weights=True) # Garde les meilleurs poids trouvés

# Sauvegarder le meilleur modèle trouvé pendant l'entraînement
model_checkpoint = ModelCheckpoint(
    filepath='best_tomato_model.h5', # Nom du fichier pour le meilleur modèle
    monitor='val_accuracy',
    save_best_only=True, # Ne sauvegarde que si la performance s'améliore
    verbose=1
)

# --- Entraînement du Modèle ---
print("Début de l'entraînement...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint] # Utiliser les callbacks
)

print("Entraînement terminé.")

# --- Sauvegarde du Modèle Final (les derniers poids, pas forcément les meilleurs) ---
# Le meilleur modèle est déjà sauvegardé par ModelCheckpoint sous 'best_tomato_model.h5'
# model.save('final_tomato_model.h5') # Optionnel : sauvegarder le tout dernier état

# --- Visualisation de l'Historique (Optionnel mais utile) ---
print("Génération des graphiques d'historique...")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc)) # Utiliser la longueur réelle de l'historique

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.suptitle('Training History')
plt.savefig('training_history.png') # Sauvegarde le graphique
plt.show() # Décommente si tu veux afficher directement

print("Graphiques sauvegardés dans 'training_history.png'")
print("Le meilleur modèle a été sauvegardé dans 'best_tomato_model.h5'")

# --- Évaluation Finale sur le Set de Test ---
print("\nÉvaluation du meilleur modèle sur le set de test...")

# Charger le MEILLEUR modèle sauvegardé par ModelCheckpoint
try:
    best_model = tf.keras.models.load_model('best_tomato_model.h5')

    # Évaluer
    loss_test, accuracy_test = best_model.evaluate(test_generator)

    print("-" * 30)
    print(f"Performance sur le set de test :")
    print(f"  -> Perte (Loss)    : {loss_test:.4f}")
    print(f"  -> Précision (Acc) : {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print("-" * 30)

except Exception as e:
    print(f"Erreur lors du chargement ou de l'évaluation du meilleur modèle : {e}")

print("Processus terminé.")