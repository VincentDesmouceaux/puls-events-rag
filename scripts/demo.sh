#!/bin/sh

set -e

IMAGE_NAME="puls-events-rag:local"
CONTAINER_NAME="puls-events-rag-demo"
HOST_PORT="8000"
CONTAINER_PORT="8000"

echo "=== Puls-Events RAG - Démonstration locale ==="

echo ""
echo "1. Vérification de Docker..."
docker info > /dev/null 2>&1 || {
  echo "Erreur : Docker ne semble pas démarré."
  exit 1
}

echo "Docker est disponible."

echo ""
echo "2. Vérification du fichier .env..."
if [ ! -f ".env" ]; then
  echo "Erreur : fichier .env introuvable à la racine du projet."
  exit 1
fi

echo ".env trouvé."

echo ""
echo "3. Arrêt de l'ancien conteneur si nécessaire..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker rm -f "${CONTAINER_NAME}" > /dev/null
  echo "Ancien conteneur supprimé."
else
  echo "Aucun ancien conteneur à supprimer."
fi

echo ""
echo "4. Construction de l'image Docker..."
docker build -t "${IMAGE_NAME}" .

echo ""
echo "5. Démarrage du conteneur..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  --env-file .env \
  "${IMAGE_NAME}"

echo ""
echo "6. Attente du démarrage de l'API..."

MAX_ATTEMPTS=30
ATTEMPT=1

while [ "${ATTEMPT}" -le "${MAX_ATTEMPTS}" ]; do
  if curl -fsS "http://127.0.0.1:${HOST_PORT}/health" > /dev/null 2>&1; then
    echo ""
    echo "API disponible."
    echo ""
    echo "Health :"
    curl -s "http://127.0.0.1:${HOST_PORT}/health"
    echo ""
    echo ""
    echo "Swagger : http://127.0.0.1:${HOST_PORT}/docs"
    echo "API     : http://127.0.0.1:${HOST_PORT}"
    echo ""
    echo "Conteneur : ${CONTAINER_NAME}"
    echo ""
    echo "=== Démonstration prête ==="
    exit 0
  fi

  echo "Tentative ${ATTEMPT}/${MAX_ATTEMPTS} - API pas encore prête..."
  sleep 2
  ATTEMPT=$((ATTEMPT + 1))
done

echo ""
echo "Erreur : l'API n'a pas répondu dans le délai prévu."
echo ""
echo "Logs du conteneur :"
docker logs "${CONTAINER_NAME}"

exit 1