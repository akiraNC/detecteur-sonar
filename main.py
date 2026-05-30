"""
main.py — API FastAPI pour la classification SONAR (Mine vs Roche)
Usage : uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Literal
import joblib
import numpy as np
import os

# ============================================================
# Chargement du modèle
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modele", "modele_sonar.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}")

modele = joblib.load(MODEL_PATH)

# ============================================================
# Schémas Pydantic
# ============================================================
class SignalSonar(BaseModel):
    frequences: list[float]

    @field_validator("frequences")
    @classmethod
    def valider_frequences(cls, v):
        if len(v) != 60:
            raise ValueError(f"Il faut exactement 60 fréquences — {len(v)} reçues.")
        if any(f < 0.0 or f > 1.0 for f in v):
            raise ValueError("Toutes les fréquences doivent être entre 0 et 1.")
        return v


class ResultatPrediction(BaseModel):
    prediction: Literal["MINE", "ROCHE"]
    label_encode: int                  # 1 = Mine, 0 = Roche
    probabilite_mine: float
    probabilite_roche: float
    confiance: float                   # probabilité de la classe prédite


# ============================================================
# Application FastAPI
# ============================================================
app = FastAPI(
    title="SONAR API — Détection Mines vs Roches",
    description="""
API de classification de signaux sonar.

## Utilisation

Envoie 60 valeurs de fréquences sonar (entre 0 et 1) et reçois :
- La **prédiction** : MINE ou ROCHE
- Les **probabilités** pour chaque classe
- Le **niveau de confiance** du modèle

## Modèle

Random Forest Classifier entraîné sur le dataset UCI SONAR
- 208 signaux d'entraînement
- 60 features (fréquences sonores)
- Accuracy : **88.1%**
    """,
    version="1.0.0",
)


# ============================================================
# Routes
# ============================================================
@app.get("/")
def accueil():
    """Point d'entrée — vérifie que l'API est en ligne."""
    return {
        "message": "SONAR API opérationnelle",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health_check():
    """Vérifie que le modèle est chargé et opérationnel."""
    return {
        "status": "ok",
        "modele_charge": modele is not None,
        "modele_type": type(modele).__name__,
    }


@app.post("/predict", response_model=ResultatPrediction)
def predict(signal: SignalSonar):
    """
    Classifie un signal sonar en Mine ou Roche.

    - **frequences** : liste de 60 valeurs float entre 0 et 1
    """
    try:
        X = np.array(signal.frequences).reshape(1, -1)

        prediction   = modele.predict(X)[0]
        probabilites = modele.predict_proba(X)[0]

        est_mine         = bool(prediction == 1)
        label            = "MINE" if est_mine else "ROCHE"
        proba_mine       = round(float(probabilites[1]), 4)
        proba_roche      = round(float(probabilites[0]), 4)
        confiance        = round(float(probabilites[prediction]), 4)

        return ResultatPrediction(
            prediction=label,
            label_encode=int(prediction),
            probabilite_mine=proba_mine,
            probabilite_roche=proba_roche,
            confiance=confiance,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")
