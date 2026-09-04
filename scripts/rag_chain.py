import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from scripts.build_faiss_index import load_faiss_index

load_dotenv()


def get_llm() -> ChatMistralAI:
    """Configure le modèle de génération Mistral."""
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is missing")

    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0.2,
    )


@lru_cache(maxsize=8)
def get_retriever(k: int = 5):
    """Recharge FAISS une seule fois par valeur de k."""
    index = load_faiss_index()

    return index.as_retriever(
        search_kwargs={"k": k},
    )


def clear_retriever_cache() -> None:
    """Vide le cache FAISS après reconstruction de l'index."""
    get_retriever.cache_clear()


def format_documents(documents) -> str:
    """Formate les événements récupérés pour le prompt."""
    formatted_documents = []

    for document in documents:
        metadata = document.metadata

        formatted_documents.append(
            (
                f"Titre : {metadata.get('title', '')}\n"
                f"Description : {metadata.get('description', '')}\n"
                f"Lieu : {metadata.get('address', '')}, "
                f"{metadata.get('city', '')}\n"
                f"Date affichée : {metadata.get('date_range', '')}\n"
                f"Début ISO : {metadata.get('start_date', '')}\n"
                f"Fin ISO : {metadata.get('end_date', '')}\n"
                f"Mots-clés : "
                f"{', '.join(metadata.get('keywords', []))}"
            )
        )

    return "\n\n---\n\n".join(formatted_documents)


def answer_question(
    question: str,
    k: int = 5,
) -> str:
    """Répond à une question à partir des événements FAISS."""
    retriever = get_retriever(k=k)

    documents = retriever.invoke(question)

    context = format_documents(documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Tu es un assistant spécialisé dans la recommandation "
                    "d'événements culturels.\n"
                    "Réponds uniquement à partir des événements présents "
                    "dans le contexte fourni.\n"
                    "N'invente jamais un événement, une date ou un lieu.\n"
                    "Pour chaque recommandation, indique l'année en t'appuyant "
                    "sur les champs Début ISO et Fin ISO du contexte.\n"
                    "N'invente ni ne modifie jamais une date.\n"
                    "Reprends fidèlement les lieux et descriptions présents "
                    "dans le contexte.\n"
                    "Si le contexte ne permet pas de répondre correctement, "
                    "indique qu'aucun événement pertinent n'a été trouvé.\n"
                    "Présente les recommandations de manière claire, "
                    "concise et naturelle."
                ),
            ),
            (
                "human",
                (
                    "Question de l'utilisateur :\n"
                    "{question}\n\n"
                    "Événements disponibles :\n"
                    "{context}"
                ),
            ),
        ]
    )

    llm = get_llm()

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return response.content


if __name__ == "__main__":
    question = "Je cherche un événement jazz à Paris."

    print(answer_question(question))