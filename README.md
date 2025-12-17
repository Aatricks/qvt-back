# QVT-Back

Backend de visualisation QVCTi. Une API sans état construite avec FastAPI et Altair pour générer des spécifications de visualisation Vega-Lite.

## Fonctionnalités

- API sans état (pas de stockage de données) conçue pour générer des spécifications de visualisation à la demande.
- Construction programmatique de spécifications JSON Vega-Lite utilisant Altair.
- Traitement efficace des données avec Pandas et NumPy.

## Prérequis

- Python 3.10 ou supérieur

## Installation

1. Créer et activer un environnement virtuel :
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sous Windows : .venv\Scripts\activate
   ```

2. Installer les dépendances :
   ```bash
   # Installer les dépendances principales
   pip install -e .

   # Installer les dépendances de développement (tests, linting)
   pip install -e ".[dev]"
   
   # Si vous utilisez uv: 
   # uv sync
   ```

## Utilisation

### Lancer le serveur API

Démarrer le serveur de développement avec rechargement automatique (hot-reload) :

```bash
uvicorn src.api.app:app --reload
```

L'API sera disponible à l'adresse `http://127.0.0.1:8000`.
- Vérification de l'état (Health Check) : `GET /health`
- Documentation : `http://127.0.0.1:8000/docs`

### Lancer les tests

Exécuter la suite de tests avec `pytest` :

```bash
pytest
```
