# Puls-Events RAG

Proof of Concept d'un assistant intelligent de recommandation d'événements culturels basé sur une architecture RAG (*Retrieval-Augmented Generation*).

## Objectif du projet

Ce projet vise à démontrer la faisabilité technique d'un chatbot capable de répondre aux questions des utilisateurs concernant des événements culturels récents.

Le système exploitera les données issues de l'API Open Agenda, une recherche vectorielle ainsi qu'un modèle de langage afin de produire des réponses contextualisées.

Le POC devra notamment permettre :

- la récupération et la préparation des données d'événements ;
- la transformation des données textuelles en embeddings ;
- l'indexation vectorielle avec FAISS ;
- la recherche sémantique des événements pertinents ;
- la génération de réponses avec un LLM Mistral ;
- l'exposition du système via une API REST ;
- l'évaluation automatisée des performances du système.

## Stack technique

- Python 3.12
- uv
- LangChain
- Mistral AI
- FAISS CPU
- Hugging Face
- Sentence Transformers
- FastAPI
- Uvicorn
- Pytest
- pytest-cov
- HTTPX
- python-dotenv

Zvec est également installé à titre expérimental.

FAISS reste la base vectorielle principale retenue pour le POC conformément au cahier des charges de Puls-Events.

## Structure du projet

```text
puls-events-rag/
├── app/
│   ├── __init__.py
│   └── main.py
│
├── scripts/
│   └── __init__.py
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   └── test_environment.py
│
├── docs/
│   └── .gitkeep
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Description des répertoires

### `app/`

Contiendra l'application principale ainsi que la future API REST FastAPI.

### `scripts/`

Contiendra les scripts nécessaires au traitement du pipeline RAG, notamment :

- récupération des données Open Agenda ;
- nettoyage des données ;
- préparation des documents ;
- génération des embeddings ;
- construction et reconstruction de l'index FAISS.

### `data/raw/`

Contiendra les données brutes récupérées depuis l'API Open Agenda.

### `data/processed/`

Contiendra les données nettoyées et préparées avant leur vectorisation.

### `tests/`

Contient les tests automatisés du projet.

Le premier test permet de vérifier que les principales dépendances nécessaires au système RAG sont correctement disponibles.

### `docs/`

Contiendra la documentation technique du projet.

## Prérequis

Python 3.12 est utilisé pour le développement du projet.

Le projet nécessite également `uv` pour la gestion de l'environnement virtuel et des dépendances.

Vérifier les versions installées :

```bash
python --version
uv --version
```

Exemple d'environnement validé :

```text
Python 3.12.12
uv 0.9.14
```

## Installation du projet

Cloner le dépôt puis se placer dans le répertoire du projet :

```bash
cd puls-events-rag
```

Synchroniser ensuite l'environnement avec `uv` :

```bash
uv sync
```

Cette commande installe les dépendances définies dans `pyproject.toml` en utilisant les versions verrouillées dans `uv.lock`.

L'environnement virtuel est créé localement dans :

```text
.venv/
```

Ce dossier ne doit pas être versionné dans Git.

## Gestion des dépendances

Le projet utilise principalement trois fichiers.

### `pyproject.toml`

Déclare les dépendances directes et la configuration du projet Python.

### `uv.lock`

Verrouille les versions exactes des dépendances et dépendances transitives.

Il permet de reproduire un environnement cohérent sur une autre machine.

### `requirements.txt`

Un fichier `requirements.txt` est également fourni afin de respecter le format traditionnel utilisé par `pip` et les exigences du livrable.

Il est généré depuis l'environnement `uv` avec :

```bash
uv export --format requirements.txt --no-hashes --output-file requirements.txt
```

## Variables d'environnement

Le projet utilisera l'API Mistral.

La clé API ne doit jamais être enregistrée directement dans le code ou versionnée dans Git.

Un modèle de configuration est fourni :

```text
.env.example
```

Créer le fichier local `.env` :

```bash
cp .env.example .env
```

Puis renseigner la clé :

```dotenv
MISTRAL_API_KEY=your_mistral_api_key_here
```

Le fichier `.env` est exclu du dépôt par `.gitignore`.

## Vérification de l'environnement

Les principales dépendances du système RAG sont testées automatiquement avec Pytest.

Le test vérifie notamment la disponibilité de :

- FAISS ;
- l'intégration FAISS de LangChain ;
- Hugging Face Embeddings ;
- le SDK Mistral.

Lancer les tests :

```bash
uv run pytest -v
```

Résultat attendu :

```text
tests/test_environment.py::test_rag_dependencies_import PASSED
```

## Imports RAG validés

L'environnement actuel utilise les imports suivants :

```python
import faiss

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from mistralai.client import Mistral
```

L'import FAISS provenant de `langchain_community` génère actuellement un avertissement de dépréciation.

Cet avertissement n'empêche pas le fonctionnement du système et sera surveillé lors des prochaines évolutions des dépendances LangChain.

## FAISS

Le projet utilise :

```text
faiss-cpu
```

plutôt que :

```text
faiss-gpu
```

Ce choix améliore la portabilité du POC et facilite son installation sur différentes machines.

## Zvec

Zvec est installé dans l'environnement à titre expérimental.

Il pourra éventuellement être utilisé pour effectuer des comparaisons avec FAISS.

Cependant, FAISS reste la technologie de référence du projet afin de respecter les exigences du cahier des charges Puls-Events.

## Tests

Les tests sont exécutés avec Pytest :

```bash
uv run pytest -v
```

Les dépendances de développement incluent notamment :

```text
pytest
pytest-cov
httpx
```

La couverture de tests pourra être exécutée ultérieurement avec :

```bash
uv run pytest --cov
```

## Reproductibilité

Pour vérifier qu'un nouvel environnement peut être reconstruit à partir des fichiers du dépôt :

```bash
uv sync
```

Puis :

```bash
uv run pytest -v
```

L'objectif est qu'une nouvelle machine puisse reproduire l'environnement sans récupérer le dossier `.venv` d'un autre développeur.

## Sécurité

Les éléments suivants ne doivent jamais être versionnés :

```text
.venv/
.env
.idea/
.DS_Store
__pycache__/
.pytest_cache/
.coverage
htmlcov/
```

Les secrets, notamment la clé API Mistral, doivent être transmis exclusivement par variables d'environnement.

## Workflow Git

Le projet utilise GitFlow.

La branche de production est :

```text
main
```

La branche d'intégration est :

```text
develop
```

Les nouvelles fonctionnalités sont développées dans des branches :

```text
feature/*
```

Pour la configuration initiale de l'environnement :

```text
feature/setup-environment
```

Le workflow général est :

```text
feature/*
    ↓
develop
    ↓
release/*
    ↓
main
```

## État actuel

L'étape de configuration de l'environnement permet actuellement de :

- utiliser Python 3.12 dans un environnement `.venv` géré avec `uv` ;
- installer les dépendances de manière reproductible ;
- importer FAISS ;
- importer LangChain ;
- importer Hugging Face Embeddings ;
- importer le SDK Mistral ;
- importer Zvec ;
- exécuter les tests avec Pytest ;
- préparer le projet pour les prochaines étapes du pipeline RAG.

## Prochaine étape

La prochaine phase du projet consistera à récupérer, analyser et préparer les données culturelles provenant de l'API Open Agenda avant leur transformation en représentations vectorielles.