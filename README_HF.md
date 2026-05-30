---
title: Detecteur Sonar
emoji: 📡
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 📡 Classification de Signaux Sonar (Mines vs Roches)

Cette application utilise un modèle **Random Forest** (Scikit-Learn) pour analyser des signaux sonar (60 fréquences) et détecter s'il s'agit d'une **Mine 💣** ou d'une **Roche 🪨**.

### 🚀 Comment ça marche ?
1. Remplis les 60 valeurs de fréquences dans l'interface.
2. L'application interroge une API FastAPI et affiche la prédiction avec son niveau de confiance.
