# Puls-Events RAG

> **Proof of Concept — Assistant intelligent de recommandation d’événements culturels**  
> Architecture RAG (*Retrieval-Augmented Generation*) basée sur **OpenAgenda**, **LangChain**, **Mistral AI**, **FAISS**, **FastAPI**, **Ragas**, **Streamlit**, **Docker** et **GitHub Actions**.

---

## Sommaire

1. [Présentation](#présentation)
2. [Objectifs du POC](#objectifs-du-poc)
3. [Architecture générale](#architecture-générale)
4. [Stack technique](#stack-technique)
5. [Structure du projet](#structure-du-projet)
6. [Installation et configuration](#installation-et-configuration)
7. [Collecte et préparation des données](#collecte-et-préparation-des-données)
8. [Embeddings et index FAISS](#embeddings-et-index-faiss)
9. [Chaîne RAG](#chaîne-rag)
10. [API REST FastAPI](#api-rest-fastapi)
11. [Reconstruction de l’index](#reconstruction-de-lindex)
12. [Évaluation avec Ragas](#évaluation-avec-ragas)
13. [Dashboard Streamlit](#dashboard-streamlit)
14. [Tests automatisés](#tests-automatisés)
15. [Docker](#docker)
16. [CI/CD et déploiement](#cicd-et-déploiement)
17. [Sécurité](#sécurité)
18. [Workflow Git](#workflow-git)
19. [Résultats du POC](#résultats-du-poc)
20. [Limites et perspectives](#limites-et-perspectives)
21. [Version actuelle](#version-actuelle)

---

## Présentation

**Puls-Events RAG** est un Proof of Concept destiné à évaluer la faisabilité d’un assistant capable de recommander des événements culturels à partir d’une question formulée en langage naturel.

Le système ne demande pas au modèle de langage de répondre uniquement à partir de ses connaissances générales. Il récupère d’abord des événements pertinents dans une base vectorielle, puis transmet ce contexte au LLM afin de générer une réponse fondée sur les données disponibles.

Le POC combine :

- les données événementielles de l’API **OpenAgenda** ;
- un pipeline de préparation et de normalisation des données ;
- les embeddings **Mistral `mistral-embed`** ;
- une base vectorielle **FAISS** ;
- un retriever **LangChain** ;
- un modèle génératif **Mistral** ;
- une API REST **FastAPI** ;
- une évaluation automatique avec **Ragas** ;
- un dashboard de suivi **Streamlit** ;
- une conteneurisation **Docker** ;
- une chaîne d’intégration et de validation avec **GitHub Actions** ;
- un déploiement du POC sur **Northflank**.

La zone géographique choisie pour cette démonstration est **Paris**.

---

## Objectifs du POC

### Objectif métier

L’objectif est de permettre à un utilisateur de poser une question telle que :

```text
Je cherche un concert de jazz à Paris.
```

et d’obtenir une réponse construite à partir des événements effectivement présents dans le corpus.

### Objectifs techniques

Le système doit permettre de :

- récupérer des événements depuis OpenAgenda ;
- nettoyer et normaliser les données ;
- filtrer les événements selon leur localisation et leur date ;
- construire une représentation textuelle exploitable ;
- découper les documents en chunks ;
- générer des embeddings ;
- créer et persister un index vectoriel FAISS ;
- effectuer une recherche sémantique ;
- construire un contexte RAG ;
- générer une réponse avec Mistral ;
- exposer le système via une API REST ;
- reconstruire la base vectorielle à la demande ;
- sécuriser l’endpoint sensible de reconstruction ;
- évaluer automatiquement le système avec Ragas ;
- superviser les métriques principales via Streamlit ;
- tester et déployer le POC de façon reproductible.

### Périmètre temporel

Les événements conservés couvrent :

- les **365 derniers jours** ;
- tous les événements futurs disponibles dans la source.

Ce choix permet de conserver un historique récent tout en privilégiant les événements encore utiles à la recommandation.

---

## Architecture générale

### Pipeline RAG

```text
OpenAgenda API
      │
      ▼
Collecte des événements
      │
      ▼
Nettoyage / normalisation
      │
      ▼
Filtrage temporel et géographique
      │
      ▼
Construction de embedding_text
      │
      ▼
Chunking
      │
      ▼
Mistral Embeddings
      │
      ▼
FAISS
      │
      ▼
Retriever LangChain
      │
      ▼
Documents pertinents
      │
      ▼
Construction du contexte
      │
      ▼
Prompt RAG
      │
      ▼
LLM Mistral
      │
      ▼
Réponse contextualisée
      │
      ▼
FastAPI / Streamlit
```

### Principe

Le RAG sépare deux opérations :

1. **Retrieval** : rechercher les documents les plus pertinents dans FAISS.
2. **Generation** : transmettre ces documents au LLM pour générer une réponse.

Cette approche permet de réduire les hallucinations et de spécialiser la génération sur les événements réellement disponibles dans le corpus.

---

## Stack technique

### Backend et API

- Python 3.12
- FastAPI
- Uvicorn
- Pydantic
- HTTPX
- python-dotenv

### RAG et Machine Learning

- LangChain
- Mistral AI
- `mistral-embed`
- `ministral-3b-latest`
- FAISS CPU
- pandas
- NumPy
- Ragas

### Interface et monitoring

- Streamlit

### Qualité et déploiement

- Pytest
- pytest-cov
- Docker
- GitHub Actions
- Northflank
- GitFlow

FAISS constitue la base vectorielle principale du POC.

---

## Structure du projet

```text
puls-events-rag/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── scripts/
│   ├── __init__.py
│   ├── fetch_openagenda.py
│   ├── build_faiss_index.py
│   ├── rag_chain.py
│   ├── evaluate_rag.py
│   └── start.sh
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── evaluation/
│   │   ├── rag_questions.json
│   │   └── ragas_results.json
│   └── faiss_index/
│       ├── index.faiss
│       └── index.pkl
│
├── tests/
│   ├── test_environment.py
│   ├── test_api.py
│   ├── test_rag_chain.py
│   └── test_evaluate_rag.py
│
├── docs/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

### Responsabilités principales

- `app/main.py` : routes FastAPI, monitoring, sécurité et orchestration du rebuild.
- `app/schemas.py` : validation des entrées et sorties de l’API avec Pydantic.
- `scripts/fetch_openagenda.py` : récupération, normalisation et préparation des événements.
- `scripts/build_faiss_index.py` : création, vérification et sauvegarde de l’index FAISS.
- `scripts/rag_chain.py` : logique métier du système RAG.
- `scripts/evaluate_rag.py` : évaluation métier et Ragas.
- `dashboard/app.py` : interface Streamlit de démonstration et de supervision.
- `tests/` : tests unitaires et fonctionnels.

---

## Installation et configuration

### Prérequis

Le projet utilise :

```text
Python 3.12.12
uv 0.9.14
```

Vérification :

```bash
python --version
uv --version
```

### Installation des dépendances

Depuis la racine du projet :

```bash
uv sync
```

Pour une installation strictement reproductible :

```bash
uv sync --frozen
```

`uv` utilise `pyproject.toml` et `uv.lock` afin de reconstruire l’environnement Python.

L’environnement virtuel est créé dans :

```text
.venv/
```

Ce dossier n’est pas versionné.

### Variables d’environnement

Créer un fichier `.env` à partir du modèle :

```bash
cp .env.example .env
```

Variables nécessaires :

```dotenv
MISTRAL_API_KEY=your_mistral_api_key_here
OPENAGENDA_API_KEY=your_openagenda_api_key_here
REBUILD_API_KEY=your_rebuild_api_key_here
```

Pour utiliser le dashboard contre l’API locale :

```dotenv
API_BASE_URL=http://127.0.0.1:8000
```

Les valeurs réelles des secrets ne doivent jamais être ajoutées au dépôt Git.

---

## Collecte et préparation des données

### Source OpenAgenda

Le corpus du POC provient de :

```text
Zone géographique : Paris
Agenda : JASS CLUB PARIS
UID OpenAgenda : 20272888
```

Le pipeline gère la pagination de l’API afin de récupérer l’ensemble des événements disponibles.

### Normalisation

Chaque événement est converti dans une structure homogène comprenant notamment :

```text
uid
title
description
city
address
latitude
longitude
start_date
end_date
date_range
keywords
status
embedding_text
```

### Champ `embedding_text`

`embedding_text` rassemble les informations utiles à la recherche sémantique :

- titre ;
- description ;
- adresse ;
- ville ;
- dates ;
- mots-clés.

### Filtrage

Les événements sont filtrés selon :

- la ville `Paris` ;
- une borne temporelle de 365 jours dans le passé ;
- l’ensemble des événements futurs disponibles.

### Chunking

Le découpage est réalisé avec `RecursiveCharacterTextSplitter`.

Configuration :

```text
chunk_size = 500
chunk_overlap = 50
```

Le chevauchement limite la perte de contexte entre deux fragments successifs.

Dans le corpus actuel, la majorité des événements tient toutefois dans un seul chunk.

---

## Embeddings et index FAISS

### Embeddings

Modèle utilisé :

```text
mistral-embed
```

Dimension :

```text
1024
```

Les embeddings transforment le contenu textuel en vecteurs numériques qui peuvent être comparés lors de la recherche sémantique.

### FAISS

L’index vectoriel est construit avec FAISS.

Pour le volume actuel du POC, une recherche exacte est suffisante.

Lors d’une exécution récente du pipeline :

```text
Événements : 310
Chunks     : 310
Vecteurs   : 310
```

Ces valeurs peuvent évoluer, car OpenAgenda est une source dynamique.

### Persistance

L’index est enregistré dans :

```text
data/faiss_index/
```

avec :

```text
index.faiss
index.pkl
```

`index.faiss` contient l’index vectoriel.

`index.pkl` contient les informations complémentaires utilisées par le VectorStore LangChain.

Le chargement du pickle n’est autorisé que pour un index généré par le projet et provenant d’une source de confiance.

---

## Chaîne RAG

### Localisation

La logique métier du système est définie dans :

```text
scripts/rag_chain.py
```

### Fonctions principales

```text
get_llm()
get_retriever()
retrieve_documents()
format_document()
format_documents()
generate_answer()
answer_question()
answer_question_with_context()
clear_retriever_cache()
```

### Fonction centrale

`answer_question()` est le point d’entrée principal du système RAG.

Son rôle est de :

1. récupérer les documents les plus pertinents ;
2. construire le contexte ;
3. transmettre ce contexte au modèle Mistral ;
4. renvoyer la réponse générée.

### Variante pour l’évaluation

`answer_question_with_context()` renvoie également les contextes récupérés.

Cette information est nécessaire pour évaluer la qualité du retrieval et de la génération avec Ragas.

### Séparation de la logique métier et de l’API

La logique RAG n’est pas implémentée directement dans FastAPI.

`app/main.py` importe et appelle les fonctions définies dans `scripts/rag_chain.py`.

Architecture :

```text
POST /ask
    │
    ▼
app/main.py
    │
    ▼
answer_question()
    │
    ▼
scripts/rag_chain.py
    │
    ├── Retriever FAISS
    ├── Construction du contexte
    └── Mistral
```

Cette séparation permet :

- de tester le RAG indépendamment de l’API ;
- de réutiliser les fonctions dans d’autres interfaces ;
- de limiter le couplage entre logique métier et transport HTTP.

### Cache du retriever

Le retriever FAISS est mis en cache pour éviter de recharger l’index à chaque requête.

Après une reconstruction de l’index, `clear_retriever_cache()` invalide ce cache afin que les requêtes suivantes utilisent immédiatement le nouvel index.

---

## Modèle de génération

Le modèle de génération utilisé est :

```text
ministral-3b-latest
```

Le modèle est appelé via l’intégration LangChain/Mistral.

Une température faible est utilisée afin de favoriser des réponses factuelles et relativement déterministes.

Le prompt demande notamment au modèle :

- de répondre uniquement à partir du contexte disponible ;
- de ne pas inventer d’événements ;
- de ne pas inventer de dates ;
- de ne pas inventer de lieux ;
- de signaler lorsqu’aucune information pertinente n’est disponible.

---

## API REST FastAPI

### Lancement local

```bash
uv run uvicorn app.main:app --reload
```

API locale :

```text
http://127.0.0.1:8000
```

Swagger :

```text
http://127.0.0.1:8000/docs
```

### `GET /health`

Vérifie l’état de l’API.

```bash
curl http://127.0.0.1:8000/health
```

Exemple :

```json
{
  "status": "ok",
  "service": "puls-events-rag-api",
  "version": "0.2.6"
}
```

### `POST /ask`

Permet d’interroger le système RAG.

```bash
curl -X POST \
  http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Je cherche un événement jazz à Paris."}'
```

Réponse :

```json
{
  "question": "Je cherche un événement jazz à Paris.",
  "answer": "Réponse générée à partir des événements retrouvés dans FAISS."
}
```

### Validation des entrées

Les données entrantes sont validées avec les schémas Pydantic définis dans :

```text
app/schemas.py
```

Les tests couvrent notamment :

- question vide ;
- champ `question` absent ;
- mauvaise requête ;
- erreur interne du RAG ;
- limitation temporaire du fournisseur LLM.

Une erreur de rate limit du fournisseur peut être traduite en :

```text
HTTP 503 Service Unavailable
```

### Endpoints complémentaires

L’API expose également :

```text
GET  /metrics
GET  /index/info
GET  /evaluation/status
POST /rebuild
GET  /rebuild/status
```

---

## Reconstruction de l’index

### `POST /rebuild`

L’endpoint permet de reconstruire l’index vectoriel sans bloquer la requête HTTP.

Il est protégé par une clé dédiée.

Header attendu :

```text
X-Rebuild-Key
```

Exemple :

```bash
curl -X POST \
  http://127.0.0.1:8000/rebuild \
  -H "X-Rebuild-Key: YOUR_REBUILD_API_KEY"
```

Réponse :

```json
{
  "status": "accepted",
  "message": "Reconstruction FAISS lancée en arrière-plan."
}
```

Code HTTP :

```text
202 Accepted
```

### Sécurité

La clé attendue est chargée depuis :

```text
REBUILD_API_KEY
```

L’API renvoie :

- `401` si la clé est absente ou incorrecte ;
- `503` si la protection n’est pas configurée ;
- `409` si un rebuild est déjà en cours.

### Background Task

FastAPI utilise `BackgroundTasks` pour exécuter la reconstruction.

```text
POST /rebuild
      │
      ▼
Validation de X-Rebuild-Key
      │
      ▼
HTTP 202
      │
      ▼
BackgroundTasks
      │
      ▼
OpenAgenda
      │
      ▼
Prétraitement
      │
      ▼
Embeddings
      │
      ▼
FAISS
      │
      ▼
Sauvegarde
      │
      ▼
Invalidation du cache
      │
      ▼
completed
```

### `GET /rebuild/status`

Permet de connaître l’état du rebuild.

```bash
curl http://127.0.0.1:8000/rebuild/status
```

Exemple :

```json
{
  "status": "completed",
  "started_at": "2026-09-12T14:00:00+00:00",
  "completed_at": "2026-09-12T14:00:20+00:00",
  "duration_seconds": 20.0,
  "error": null,
  "progress": 100,
  "step": "Reconstruction FAISS terminée",
  "processed": 310,
  "total": 310
}
```

### États possibles

```text
idle
running
completed
failed
```

### Progression du rebuild

La version `0.2.6` expose une progression structurée :

```text
5 %   Initialisation
10 %  Récupération OpenAgenda
25 %  Conversion des événements
35 %  Filtrage
45 %  Préparation des textes
55 %  Chunking
65 %  Embeddings + construction FAISS
90 %  Vérification
95 %  Sauvegarde
100 % Reconstruction terminée
```

La progression correspond aux phases réellement exécutées.

La génération des embeddings est effectuée en bloc par le VectorStore FAISS ; le pourcentage peut donc rester momentanément à `65 %` pendant cette étape.

### Limites de l’implémentation

Cette solution est adaptée à un POC.

L’état du rebuild est conservé en mémoire dans le processus FastAPI. Par conséquent :

- un redémarrage du conteneur réinitialise cet état ;
- un redémarrage pendant le traitement peut interrompre le rebuild ;
- plusieurs replicas ne partageraient pas automatiquement le même état.

Une industrialisation pourrait externaliser cette tâche dans un worker dédié avec une file de jobs et un stockage persistant.

---

## Évaluation avec Ragas

### Jeu d’évaluation

Les scénarios sont définis dans :

```text
data/evaluation/rag_questions.json
```

Le dataset contient actuellement **7 scénarios**, comprenant :

- des recherches jazz ;
- une jam session ;
- une soirée swing ;
- du jazz manouche ;
- une recherche croisant jazz et cinéma ;
- une recherche basée sur une adresse ;
- un scénario négatif hors corpus.

### Script d’évaluation

```text
scripts/evaluate_rag.py
```

Lancement complet :

```bash
uv run python -m scripts.evaluate_rag
```

Lancement limité :

```bash
uv run python -m scripts.evaluate_rag --limit 1
```

### Métriques Ragas

Trois métriques sont utilisées :

#### Faithfulness

Mesure si la réponse générée est bien soutenue par le contexte récupéré.

#### Response Relevancy

Mesure la pertinence de la réponse par rapport à la question utilisateur.

#### LLM Context Recall

Mesure si le contexte récupéré contient les informations nécessaires à la réponse attendue.

### Résultats de référence

Dernière évaluation complète sur 7 scénarios :

```text
Faithfulness       ≈ 0.780
ResponseRelevancy  ≈ 0.712
LLMContextRecall   ≈ 0.857
```

Les six scénarios positifs ont obtenu :

```text
Context Recall = 1.0
```

Le scénario négatif obtient des métriques automatiques faibles alors que le comportement métier est correct : le système refuse d’inventer un événement qui n’existe pas dans le corpus.

Ce résultat illustre une limite importante des métriques automatiques : elles doivent être interprétées avec une évaluation métier et non utilisées isolément.

### Résultats persistés

Les résultats sont enregistrés dans :

```text
data/evaluation/ragas_results.json
```

Le fichier contient notamment :

- la date de génération ;
- les scores globaux ;
- les scores par scénario ;
- la réponse produite ;
- la réponse de référence ;
- les métriques Ragas.

---

## Dashboard Streamlit

### Objectif

Le dashboard apporte une interface de démonstration et de supervision du POC.

Fichier :

```text
dashboard/app.py
```

### Lancement

```bash
uv run streamlit run dashboard/app.py
```

### Fonctionnalités

Le dashboard permet de visualiser :

- l’état de l’API ;
- la version déployée ;
- le nombre de requêtes ;
- les temps de réponse ;
- les erreurs ;
- les informations sur l’index FAISS ;
- le statut du rebuild ;
- la progression du rebuild ;
- les résultats Ragas ;
- un assistant culturel connecté à `/ask`.

### Onglets

```text
Vue d'ensemble
Assistant culturel
Évaluation Ragas
```

### API locale ou distante

Pour utiliser l’API locale :

```dotenv
API_BASE_URL=http://127.0.0.1:8000
```

Le dashboard peut également cibler l’API Northflank en utilisant son URL de production.

---

## Tests automatisés

### Exécution

```bash
uv run pytest -q
```

État validé pour la version `0.2.6` :

```text
38 passed
```

Un avertissement de dépréciation Starlette/AnyIO peut apparaître. Il n’empêche pas l’exécution du POC.

### Couverture fonctionnelle

Les tests couvrent notamment :

- environnement Python ;
- imports et dépendances ;
- récupération OpenAgenda ;
- normalisation des événements ;
- données manquantes ;
- pagination ;
- filtrage temporel ;
- filtrage géographique ;
- construction des embeddings ;
- création des documents ;
- chaîne RAG ;
- appel du retriever ;
- génération de réponse ;
- `/health` ;
- `/ask` ;
- validation Pydantic ;
- question vide ;
- mauvaise requête ;
- erreurs du RAG ;
- rate limit ;
- protection de `/rebuild` ;
- clé absente ;
- clé incorrecte ;
- rebuild en arrière-plan ;
- erreur pendant le rebuild ;
- rebuild concurrent ;
- statut et progression du rebuild ;
- préparation du dataset Ragas ;
- logique de l’évaluation.

---

## Docker

### Construction

```bash
docker build -t puls-events-rag:local .
```

### Exécution

```bash
docker run --rm \
  --name puls-events-rag-demo \
  -p 8000:8000 \
  --env-file .env \
  puls-events-rag:local
```

L’API est ensuite disponible sur :

```text
http://127.0.0.1:8000
```

Swagger :

```text
http://127.0.0.1:8000/docs
```

### Démarrage du conteneur

Le script :

```text
scripts/start.sh
```

vérifie la présence de l’index FAISS.

Si l’index n’est pas disponible, il déclenche sa reconstruction avant le démarrage d’Uvicorn.

---

## CI/CD et déploiement

### GitHub Actions

Workflow :

```text
.github/workflows/ci.yml
```

### Validation sur les branches

La CI est exécutée sur les branches et Pull Requests concernées.

Chaîne principale :

```text
Pytest
   │
   ▼
Docker build
   │
   ▼
Docker smoke test
```

### Validation de production

Sur un push vers `main`, le workflow peut ensuite poursuivre avec :

```text
Push main
   │
   ▼
Pytest
   │
   ▼
Docker build
   │
   ▼
Docker smoke test
   │
   ▼
Déploiement Northflank
   │
   ▼
Attente de la version attendue
   │
   ▼
GET /health
   │
   ▼
POST /rebuild
   │
   ▼
GET /rebuild/status
   │
   ▼
Validation de FAISS
   │
   ▼
POST /ask
```

### Northflank

Le service de production est construit depuis :

```text
main
```

et utilise le `Dockerfile` du projet.

Endpoints principaux exposés :

```text
GET  /health
POST /ask
POST /rebuild
GET  /rebuild/status
GET  /metrics
GET  /index/info
GET  /evaluation/status
```

Le health check permet également au workflow CI/CD de vérifier que la bonne version est effectivement déployée avant de poursuivre les tests de production.

---

## Sécurité

### Secrets

Les secrets sont fournis par variables d’environnement :

```text
MISTRAL_API_KEY
OPENAGENDA_API_KEY
REBUILD_API_KEY
```

Ils ne doivent jamais être stockés dans :

- le code ;
- le README ;
- les commits ;
- les logs publics ;
- le workflow GitHub Actions en clair.

### Fichiers exclus du dépôt

Exemples :

```text
.env
.venv/
.idea/
.DS_Store
__pycache__/
.pytest_cache/
.coverage
htmlcov/
```

### Endpoint sensible

`POST /rebuild` est volontairement protégé par le header :

```text
X-Rebuild-Key
```

Cette mesure évite qu’un utilisateur anonyme puisse déclencher une opération coûteuse de reconstruction de l’index.

---

## Workflow Git

Le projet utilise GitFlow.

```text
feature/*
    │
    ▼
develop
    │
    ▼
release/*
    │
    ▼
main
```

### `develop`

Branche d’intégration des fonctionnalités terminées.

### `feature/*`

Branches utilisées pour développer une fonctionnalité isolée.

### `release/*`

Branches utilisées pour préparer et valider une version.

### `main`

Branche de production.

Les versions finalisées sont taguées afin de conserver des jalons reproductibles.

---

## Résultats du POC

Le projet démontre une chaîne complète allant de la donnée brute jusqu’à une API et une interface utilisateur.

```text
OpenAgenda
    +
Prétraitement
    +
Chunking
    +
Mistral Embeddings
    +
FAISS
    +
LangChain Retriever
    +
Mistral LLM
    +
FastAPI
    +
Ragas
    +
Streamlit
    +
Docker
    +
CI/CD
```

### Fonctionnalités opérationnelles

```text
Collecte OpenAgenda             OK
Normalisation                   OK
Filtrage                        OK
Chunking                        OK
Embeddings Mistral              OK
Index FAISS                     OK
Retriever LangChain             OK
Chaîne RAG                      OK
API FastAPI                     OK
Validation des entrées          OK
Gestion des erreurs             OK
Sécurité /rebuild               OK
Background rebuild FAISS        OK
Progression du rebuild          OK
Évaluation métier               OK
Évaluation Ragas                OK
Dashboard Streamlit             OK
Tests automatisés               OK
Docker                          OK
GitHub Actions                  OK
Déploiement Northflank          OK
```

Le POC est donc exploitable pour une démonstration de bout en bout et pour la soutenance du projet.

---

## Limites et perspectives

Le projet reste volontairement un Proof of Concept.

### Limites actuelles

- corpus limité à une zone et un agenda principal ;
- état du rebuild conservé en mémoire ;
- index FAISS non persistant sur un volume distant ;
- reconstruction exécutée dans le processus de l’API ;
- métriques Ragas parfois peu adaptées aux scénarios négatifs ;
- volume de données encore faible ;
- génération dépendante des quotas du fournisseur LLM.

### Évolutions possibles

- intégrer plusieurs agendas ;
- élargir la couverture géographique ;
- ajouter des filtres structurés sur les dates et lieux ;
- comparer plusieurs stratégies de retrieval ;
- tester plusieurs tailles de `k` ;
- ajouter du reranking ;
- augmenter le jeu d’évaluation ;
- exécuter Ragas dans un workflow dédié ;
- persister FAISS sur un stockage durable ;
- externaliser le rebuild vers un worker ;
- ajouter une base de suivi des requêtes ;
- améliorer l’observabilité ;
- suivre les coûts d’inférence ;
- automatiser les mises à jour périodiques du corpus.

---

## Version actuelle

```text
0.2.6
```

### Principales évolutions de la version `0.2.6`

- intégration de **Ragas** ;
- ajout des métriques `Faithfulness`, `ResponseRelevancy` et `LLMContextRecall` ;
- ajout de `ragas_results.json` ;
- ajout du dashboard **Streamlit** ;
- ajout de l’endpoint `/evaluation/status` ;
- suivi du rebuild FAISS par étapes ;
- ajout des champs `progress`, `step`, `processed` et `total` ;
- affichage de la progression dans le dashboard ;
- extension des tests automatisés ;
- alignement de la version API, Docker, CI/CD et documentation.

---

## Conclusion

Puls-Events RAG démontre qu’il est possible de construire un assistant culturel spécialisé en combinant recherche vectorielle et génération augmentée.

Le système ne se limite pas à une démonstration du modèle de langage : il couvre tout le cycle technique du POC, depuis la collecte des données OpenAgenda jusqu’à leur indexation, leur interrogation, l’exposition du service via FastAPI, son évaluation avec Ragas, sa supervision avec Streamlit et son déploiement conteneurisé.

L’architecture reste volontairement simple et explicable, tout en séparant clairement la logique métier RAG de la couche API. Elle constitue ainsi une base cohérente pour une démonstration, une soutenance et une future industrialisation.
