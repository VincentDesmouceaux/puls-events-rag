# Puls-Events RAG

Proof of Concept d'un assistant intelligent de recommandation d'événements culturels basé sur une architecture RAG (*Retrieval-Augmented Generation*).

Le projet combine les données de l'API OpenAgenda, des embeddings Mistral, une base vectorielle FAISS, LangChain, un modèle de langage Mistral et une API REST FastAPI.

---

## Objectif du projet

Puls-Events souhaite évaluer la faisabilité technique d'un assistant capable de répondre en langage naturel à des questions concernant des événements culturels.

Le système doit notamment permettre :

- de récupérer des événements depuis l'API OpenAgenda ;
- de nettoyer et normaliser les données ;
- de filtrer les événements selon leur localisation et leur date ;
- de transformer les données textuelles en embeddings ;
- de construire un index vectoriel FAISS ;
- d'effectuer une recherche sémantique ;
- de transmettre le contexte pertinent à un LLM ;
- de générer une réponse contextualisée ;
- d'exposer le système via une API REST ;
- d'évaluer automatiquement le comportement du système ;
- de reconstruire l'index FAISS sans bloquer l'API ;
- de déployer et valider le POC dans un environnement conteneurisé.

La zone géographique retenue pour le POC est **Paris**.

Les événements considérés couvrent :

- les 365 derniers jours ;
- l'ensemble des événements futurs disponibles.

---

## Architecture RAG

Le pipeline général du projet est le suivant :

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
Construction du texte documentaire
      │
      ▼
Découpage en chunks
      │
      ▼
Mistral Embeddings
      │
      ▼
FAISS
      │
      ▼
Recherche sémantique
      │
      ▼
Retriever LangChain
      │
      ▼
Prompt enrichi avec le contexte
      │
      ▼
LLM Mistral
      │
      ▼
Réponse utilisateur
      │
      ▼
FastAPI
```

Le principe du RAG consiste à rechercher d'abord les informations pertinentes dans une base documentaire, puis à fournir ces informations au modèle de langage afin de générer une réponse fondée sur le contexte récupéré.

Cette approche permet de limiter les hallucinations et de spécialiser les réponses du modèle sur les événements disponibles dans le corpus Puls-Events.

---

## Stack technique

Le projet utilise principalement :

- Python 3.12 ;
- uv ;
- FastAPI ;
- Uvicorn ;
- LangChain ;
- Mistral AI ;
- FAISS CPU ;
- pandas ;
- NumPy ;
- Pytest ;
- pytest-cov ;
- HTTPX ;
- python-dotenv ;
- Docker ;
- GitHub Actions ;
- Northflank.

Zvec est également présent à titre expérimental.

FAISS reste la base vectorielle principale retenue pour le POC.

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
├── data/
│   ├── raw/
│   ├── processed/
│   ├── evaluation/
│   │   └── rag_questions.json
│   └── faiss_index/
│       ├── index.faiss
│       └── index.pkl
│
├── tests/
│   ├── test_environment.py
│   ├── test_api.py
│   └── ...
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

Les fichiers générés dans `data/processed/` et `data/faiss_index/` peuvent être exclus du versionnement Git selon leur nature.

---

## Installation

### Prérequis

Le projet utilise Python 3.12 et `uv`.

Vérifier les versions installées :

```bash
python --version
uv --version
```

Environnement utilisé pendant le développement :

```text
Python 3.12.12
uv 0.9.14
```

### Installation des dépendances

Depuis la racine du projet :

```bash
uv sync
```

`uv` utilise `pyproject.toml` et `uv.lock` afin de reconstruire un environnement reproductible.

L'environnement virtuel est créé dans :

```text
.venv/
```

Ce répertoire n'est pas versionné.

---

## Variables d'environnement

Créer un fichier `.env` local à partir du modèle :

```bash
cp .env.example .env
```

Les secrets doivent ensuite être renseignés uniquement dans `.env` ou dans le gestionnaire de secrets de la plateforme de déploiement.

Exemple :

```dotenv
MISTRAL_API_KEY=your_mistral_api_key_here
OPENAGENDA_API_KEY=your_openagenda_api_key_here
REBUILD_API_KEY=your_rebuild_api_key_here
```

Le fichier `.env` ne doit jamais être versionné.

Les clés réelles ne doivent pas apparaître dans :

- le code source ;
- le README ;
- les commits Git ;
- les workflows GitHub Actions ;
- les logs publics.

---

## Collecte des données OpenAgenda

Les événements utilisés par le POC sont récupérés depuis l'API OpenAgenda.

Configuration utilisée :

```text
Zone géographique : Paris
Agenda : JASS CLUB PARIS
UID OpenAgenda : 20272888
```

La récupération prend en charge la pagination de l'API afin de parcourir l'ensemble des événements disponibles.

Le pipeline utilise un `pandas.DataFrame` pour les étapes de préparation et de manipulation des données.

---

## Filtrage temporel et géographique

Les événements sont filtrés selon :

- la ville : `Paris` ;
- les 365 derniers jours ;
- tous les événements futurs disponibles.

Les dates sont normalisées afin de garantir un filtrage temporel cohérent.

Ce choix permet de conserver à la fois un historique récent et les événements à venir pouvant être recommandés aux utilisateurs.

---

## Normalisation des événements

Chaque événement est transformé dans une structure homogène contenant notamment :

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

Le champ `embedding_text` constitue la représentation textuelle utilisée pour la vectorisation.

Il regroupe les informations utiles à la recherche sémantique, notamment :

- le titre ;
- la description ;
- le lieu ;
- les dates ;
- les mots-clés.

---

## Chunking

Avant l'indexation, les documents peuvent être découpés avec `RecursiveCharacterTextSplitter`.

Configuration utilisée :

```text
chunk_size = 500
chunk_overlap = 50
```

Le chevauchement permet de limiter la perte de contexte entre deux fragments successifs.

Dans le corpus actuellement utilisé, la majorité des événements possède toutefois une représentation suffisamment courte pour tenir dans un seul chunk.

---

## Embeddings

La vectorisation est réalisée avec Mistral AI.

Modèle :

```text
mistral-embed
```

Dimension des vecteurs :

```text
1024
```

Les embeddings transforment les descriptions textuelles des événements en représentations numériques permettant d'effectuer une recherche par proximité sémantique.

Lors d'une exécution de référence du pipeline, environ **306 événements** ont été récupérés, filtrés et vectorisés.

Ce nombre peut évoluer puisque les données OpenAgenda sont dynamiques.

---

## Base vectorielle FAISS

Le projet utilise FAISS comme moteur de recherche vectorielle.

L'index repose sur une recherche exacte adaptée à la taille actuelle du POC.

Les documents LangChain associés aux vecteurs conservent les métadonnées nécessaires à l'identification des événements.

L'index est sauvegardé localement dans :

```text
data/faiss_index/
```

avec notamment :

```text
index.faiss
index.pkl
```

`index.faiss` contient l'index vectoriel.

`index.pkl` contient les informations complémentaires nécessaires à la reconstruction du VectorStore LangChain.

Le chargement du fichier pickle doit uniquement être effectué avec un index généré par le projet et provenant d'une source de confiance.

---

## Chaîne RAG

La chaîne RAG est définie dans :

```text
scripts/rag_chain.py
```

Elle réalise les opérations suivantes :

```text
Question utilisateur
        │
        ▼
Embedding de la question
        │
        ▼
Recherche FAISS
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
Mistral
        │
        ▼
Réponse
```

Le retriever FAISS est mis en cache afin d'éviter de recharger l'index à chaque requête.

Après une reconstruction réussie de FAISS, ce cache est invalidé afin que les requêtes suivantes utilisent le nouvel index.

---

## Modèle de génération

Le modèle de génération utilisé est :

```text
ministral-3b-latest
```

Il est appelé via l'intégration LangChain/Mistral.

Une température faible est utilisée afin de favoriser des réponses relativement déterministes et adaptées à un système de recommandation fondé sur des données factuelles.

Le prompt demande au modèle :

- de répondre à partir du contexte fourni ;
- de ne pas inventer d'événements ;
- de ne pas inventer de dates ou de lieux ;
- de signaler lorsqu'aucune information pertinente n'est disponible.

---

## API REST FastAPI

Le système RAG est exposé via FastAPI.

Lancer l'API localement :

```bash
uv run uvicorn app.main:app --reload
```

Documentation Swagger :

```text
http://127.0.0.1:8000/docs
```

---

## Endpoint `/health`

Permet de vérifier l'état de l'API.

```bash
curl http://127.0.0.1:8000/health
```

Exemple de réponse :

```json
{
  "status": "ok",
  "service": "puls-events-rag-api",
  "version": "0.2.3"
}
```

---

## Endpoint `/ask`

Permet d'interroger le système RAG.

Exemple :

```bash
curl -X POST \
  http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Je cherche un événement jazz à Paris."}'
```

Structure de réponse :

```json
{
  "question": "Je cherche un événement jazz à Paris.",
  "answer": "Réponse générée à partir des événements retrouvés dans FAISS."
}
```

L'endpoint valide les données d'entrée et gère également certaines erreurs provenant du service de génération.

Une limitation temporaire du fournisseur LLM peut notamment être convertie en réponse HTTP `503 Service Unavailable`.

---

## Reconstruction de l'index FAISS

L'API fournit un endpoint permettant de déclencher une reconstruction de l'index :

```text
POST /rebuild
```

Cet endpoint est protégé par la variable :

```text
REBUILD_API_KEY
```

La clé est transmise dans le header :

```text
X-Rebuild-Key
```

Exemple :

```bash
curl -X POST \
  http://127.0.0.1:8000/rebuild \
  -H "X-Rebuild-Key: YOUR_REBUILD_API_KEY"
```

Une requête valide retourne :

```text
HTTP 202 Accepted
```

avec une réponse de type :

```json
{
  "status": "accepted",
  "message": "Reconstruction FAISS lancée en arrière-plan."
}
```

---

## Background Task

La reconstruction FAISS est exécutée avec les `BackgroundTasks` de FastAPI.

Cela permet à `/rebuild` de répondre immédiatement en HTTP `202` sans maintenir la requête HTTP ouverte pendant toute la reconstruction.

Le fonctionnement est :

```text
POST /rebuild
      │
      ▼
Validation X-Rebuild-Key
      │
      ▼
HTTP 202 Accepted
      │
      ▼
BackgroundTasks
      │
      ▼
Collecte OpenAgenda
      │
      ▼
Embeddings
      │
      ▼
Nouvel index FAISS
      │
      ▼
Invalidation du cache Retriever
      │
      ▼
status = completed
```

Une protection empêche également le lancement simultané de plusieurs reconstructions dans le même processus.

---

## Endpoint `/rebuild/status`

L'état de la reconstruction peut être consulté avec :

```bash
curl http://127.0.0.1:8000/rebuild/status
```

Les principaux états sont :

```text
idle
running
completed
failed
```

Exemple :

```json
{
  "status": "completed",
  "started_at": "2026-09-10T12:00:00+00:00",
  "completed_at": "2026-09-10T12:00:20+00:00",
  "error": null
}
```

En cas d'échec, le statut devient `failed` et une information d'erreur est conservée.

---

## Limites du Background Task

L'implémentation actuelle est adaptée à un POC.

Le traitement de fond est exécuté dans le même environnement que l'API FastAPI.

Par conséquent :

- un redémarrage du conteneur pendant la reconstruction peut interrompre le traitement ;
- l'état du rebuild est conservé en mémoire ;
- cet état est réinitialisé lors d'un redémarrage ;
- plusieurs workers ou plusieurs replicas ne partageraient pas automatiquement cet état.

Dans une architecture de production à plus grande échelle, le traitement pourrait être externalisé vers un système de jobs ou une file de tâches avec stockage persistant.

---

## Tests automatisés

Les tests sont exécutés avec Pytest.

```bash
uv run pytest -q
```

État validé pour la version `0.2.3` :

```text
33 passed
```

Les tests couvrent notamment :

- l'environnement Python ;
- les dépendances RAG ;
- la normalisation OpenAgenda ;
- les données manquantes ;
- le filtrage géographique ;
- le filtrage temporel ;
- la pagination OpenAgenda ;
- la génération des embeddings ;
- la construction des documents ;
- l'API `/health` ;
- l'API `/ask` ;
- la validation des entrées ;
- la gestion des erreurs RAG ;
- la protection de `/rebuild` ;
- le déclenchement du rebuild ;
- les erreurs du rebuild ;
- le statut de reconstruction ;
- la prévention des reconstructions concurrentes.

Deux avertissements de dépréciation peuvent actuellement apparaître dans l'environnement de test.

Ils ne bloquent pas l'exécution du POC.

---

## Évaluation du système RAG

Un jeu de questions d'évaluation est disponible dans :

```text
data/evaluation/rag_questions.json
```

Il contient plusieurs scénarios permettant de contrôler le comportement du système, notamment :

- des recherches d'événements ;
- des contraintes de localisation ;
- des demandes correspondant au corpus ;
- des demandes ne correspondant pas au corpus.

Le script :

```text
scripts/evaluate_rag.py
```

permet d'exécuter l'évaluation.

Une exécution de référence du jeu d'évaluation a obtenu :

```text
7 / 7 scénarios validés
Score : 1.00
```

Cette évaluation constitue une validation fonctionnelle du POC et non un benchmark exhaustif de la qualité sémantique d'un système RAG industriel.

---

## Docker

Le projet est conteneurisé avec Docker.

Construire l'image :

```bash
docker build -t puls-events-rag:local .
```

Lancer le conteneur :

```bash
docker run --rm \
  --name puls-events-rag-demo \
  -p 8000:8000 \
  --env-file .env \
  puls-events-rag:local
```

L'API est ensuite accessible sur :

```text
http://127.0.0.1:8000
```

et Swagger sur :

```text
http://127.0.0.1:8000/docs
```

Le script :

```text
scripts/start.sh
```

vérifie la présence de l'index FAISS au démarrage.

Si l'index n'existe pas dans le conteneur, une reconstruction est déclenchée avant le lancement d'Uvicorn.

---

## CI/CD avec GitHub Actions

Le projet dispose d'un workflow GitHub Actions dans :

```text
.github/workflows/ci.yml
```

La CI est exécutée sur les branches principales et les Pull Requests concernées.

Le pipeline valide notamment :

```text
Pytest
   │
   ▼
Docker build
   │
   ▼
Docker smoke test
```

Lors d'une mise à jour de `main`, le workflow peut également valider le déploiement de production.

Le processus cible est :

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
Vérification /health
   │
   ▼
POST /rebuild
   │
   ▼
Suivi /rebuild/status
   │
   ▼
Validation du nouvel index FAISS
   │
   ▼
Requête /ask en production
```

Les secrets nécessaires au workflow sont stockés dans les GitHub Actions Secrets et ne doivent jamais être écrits directement dans le fichier YAML.

---

## Déploiement Northflank

Le POC est déployé sur Northflank à partir de la branche :

```text
main
```

Le déploiement utilise le `Dockerfile` du projet.

L'API de production expose notamment :

```text
GET  /health
POST /ask
POST /rebuild
GET  /rebuild/status
```

Le health check permet également au pipeline de vérifier que la version attendue de l'API est effectivement déployée avant de lancer les validations de production.

---

## Reproductibilité

Pour reconstruire l'environnement :

```bash
uv sync --frozen
```

Puis exécuter les tests :

```bash
uv run pytest -q
```

Il n'est pas nécessaire de récupérer le dossier `.venv` d'une autre machine.

`pyproject.toml` décrit les dépendances du projet tandis que `uv.lock` verrouille les versions nécessaires à la reproductibilité.

---

## Sécurité

Les éléments sensibles ou locaux ne doivent pas être versionnés.

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

Les secrets sont injectés via variables d'environnement.

En particulier :

```text
MISTRAL_API_KEY
OPENAGENDA_API_KEY
REBUILD_API_KEY
```

Aucune valeur réelle de ces secrets ne doit être stockée dans le dépôt.

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

`develop` constitue la branche d'intégration.

`main` constitue la branche de production.

Les fonctionnalités sont développées dans des branches `feature/*`, puis intégrées dans `develop`.

Une branche `release/*` permet ensuite de préparer et valider une nouvelle version avant son intégration dans `main`.

---

## Version actuelle

Version du projet :

```text
0.2.3
```

Cette version introduit notamment la gestion asynchrone de la reconstruction FAISS et son intégration au workflow de validation de production.

---

## Résultats du POC

Le POC démontre qu'il est possible de construire une chaîne complète de recommandation d'événements basée sur une architecture RAG :

```text
OpenAgenda
    +
Prétraitement
    +
Mistral Embeddings
    +
FAISS
    +
LangChain
    +
Mistral LLM
    +
FastAPI
    +
Docker
```

Le système est capable de récupérer un corpus événementiel, l'indexer, rechercher les événements sémantiquement pertinents et générer une réponse contextualisée.

L'architecture permet également de reconstruire l'index lorsque les données OpenAgenda évoluent.

---

## Limites et perspectives

Le projet reste un Proof of Concept.

Les principales évolutions possibles sont :

- augmenter la couverture géographique et le nombre d'agendas ;
- enrichir les métadonnées utilisées par le retriever ;
- ajouter des filtres structurés sur les dates et les lieux ;
- comparer plusieurs stratégies de retrieval ;
- mesurer plus finement la qualité sémantique des réponses ;
- étendre le jeu d'évaluation ;
- mettre en place un stockage persistant de l'index ;
- externaliser les reconstructions longues vers un système de jobs ;
- ajouter une observabilité plus complète ;
- suivre les temps de réponse et les coûts d'inférence ;
- mettre en place une stratégie de mise à jour périodique des événements.

Pour une industrialisation, la persistance de l'index et l'exécution des tâches de reconstruction hors du processus FastAPI constitueraient deux évolutions importantes.

---

## État du projet

Les principales briques du POC sont désormais opérationnelles :

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
Tests automatisés               OK
Évaluation RAG                  OK
Docker                          OK
CI GitHub Actions               OK
Déploiement Northflank          OK
Background rebuild FAISS        OK
Suivi du rebuild                OK
```

Le projet est ainsi prêt pour la phase finale de documentation, démonstration et présentation du POC.