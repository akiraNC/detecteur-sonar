FROM python:3.12-slim

# Répertoire de travail
WORKDIR /app

# Copie des dépendances et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code et du modèle
COPY app.py .
COPY main.py .
COPY modele/ modele/

# Ports exposés : 7860 (Gradio) + 8000 (FastAPI)
EXPOSE 7860
EXPOSE 8000

# Script de démarrage : lance FastAPI + Gradio ensemble
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & python app.py"]