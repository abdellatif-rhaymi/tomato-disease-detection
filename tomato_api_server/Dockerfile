# Utilise une image Python officielle
FROM python:3.9-slim

# Définit le port que l'application écoutera (doit correspondre au code Flask)
ENV PORT 8080

# Définit le répertoire de travail
WORKDIR /app

# Copie les fichiers de dépendances et installe les dépendances
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste du code de l'application
COPY . .

# Commande pour exécuter l'application Flask
# Gunicorn est souvent utilisé en production, mais Flask dev server peut suffire pour commencer
# CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]
CMD ["python", "main.py"]