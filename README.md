---
title: Detecteur Sonar
emoji: 📡
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 📡 SONAR Classifier — Détection Mines vs Roches

Classifieur de signaux sonar basé sur **Random Forest**.  
Envoie 60 fréquences sonores → reçois une prédiction **Mine 💣** ou **Roche 🪨** avec un niveau de confiance.

[![Hugging Face](https://img.shields.io/badge/🤗%20HF%20Space-detecteur--sonar-blue)](https://huggingface.co/spaces/psychoc4t/detecteur-sonar)
[![GitHub](https://img.shields.io/badge/GitHub-akiraNC%2Fdetecteur--sonar-black)](https://github.com/akiraNC/detecteur-sonar)

---

## 🏗️ Architecture MLOps


```

Entraînement (Colab)
↓ dvc push
Google Drive (dataset + modèle versionnés)
↓ git push
GitHub (code + pointeurs .dvc)
↓
GitHub Actions (CI/CD)
├── git push code ────────> Hugging Face Space
└── hf_hub.upload_file ──> modele/modele_sonar.pkl
↓
Hugging Face Space (FastAPI + Gradio)
↓ Evidently AI
Monitoring (data drift + infra)

```

---

## 🧰 Stack technique

| Composant | Outil |
|-----------|-------|
| Modèle | Random Forest (scikit-learn) — Accuracy **88.1%** |
| API | FastAPI — endpoint `/predict` |
| Interface | Gradio |
| Conteneur | Docker |
| Versionning données | DVC + Google Drive |
| Tracking expériences | MLflow |
| CI/CD | GitHub Actions → Hugging Face Spaces |
| Monitoring | Evidently AI |

---

## 🚀 Lancer en local

### Sans Docker

```bash
# 1. Créer et activer l'environnement
mamba create -n venv
mamba activate venv
mamba install -c conda-forge python=3.12 gradio scikit-learn numpy joblib
pip install fastapi uvicorn requests

# 2. Lancer FastAPI (terminal 1)
uvicorn main:app --reload

# 3. Lancer Gradio (terminal 2)
python app.py

# 4. Ouvrir
# Interface Gradio  → http://localhost:7860
# Swagger FastAPI   → http://localhost:8000/docs

```

### Avec Docker

```bash
# 1. Builder l'image
docker build -t sonar-app .

# 2. Lancer le container (FastAPI + Gradio ensemble)
docker run -p 7860:7860 -p 8000:8000 sonar-app

# 3. Ouvrir
# Interface Gradio  → http://localhost:7860
# Swagger FastAPI   → http://localhost:8000/docs

```

---

## 📁 Structure du projet

```
├── app.py                      # Interface Gradio
├── main.py                     # API FastAPI
├── Dockerfile                  # Conteneur Docker
├── requirements.txt            # Dépendances Python
├── README.md                   # Documentation
├── .github/
│   └── workflows/
│       └── deploy.yml          # Pipeline CI/CD
├── modele/
│   └── modele_sonar.pkl.dvc    # Pointeur DVC (modèle sur Google Drive)
└── sonar.all-data.csv.dvc      # Pointeur DVC (dataset sur Google Drive)

```

---

## 🔌 API FastAPI

### `GET /health`

Vérifie que l'API et le modèle sont opérationnels.

### `POST /predict`

**Requête :**

```json
{
  "frequences": [0.02, 0.037, 0.042, ..., 0.003]
}

```

**Réponse :**

```json
{
  "prediction": "ROCHE",
  "label_encode": 0,
  "probabilite_mine": 0.12,
  "probabilite_roche": 0.88,
  "confiance": 0.88
}

```

> 📖 Documentation interactive : `http://localhost:8000/docs`

---

## 📊 Dataset

**UCI SONAR Mines vs Rocks**

* 208 signaux sonar
* 60 fréquences sonores (valeurs entre 0 et 1)
* 111 mines / 97 roches
* Source : [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks)
