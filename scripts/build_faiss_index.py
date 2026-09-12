import os
from collections.abc import Callable

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

from scripts.fetch_openagenda import (
    add_embedding_text,
    chunk_events,
    events_to_dataframe,
    fetch_all_events,
    filter_events,
)


load_dotenv()


ProgressCallback = Callable[[int, str, int | None, int | None], None]


def get_embeddings() -> MistralAIEmbeddings:
    """Configure le modèle d'embeddings Mistral."""

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is missing")

    return MistralAIEmbeddings(
        model="mistral-embed",
        api_key=api_key,
    )


def build_faiss_index(chunks: list[dict]) -> FAISS:
    """Construit un index FAISS à partir des chunks d'événements."""

    embeddings = get_embeddings()

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "uid": chunk["uid"],
                "title": chunk["title"],
                "description": chunk["description"],
                "city": chunk["city"],
                "address": chunk["address"],
                "start_date": str(chunk["start_date"]),
                "end_date": str(chunk["end_date"]),
                "date_range": chunk["date_range"],
                "keywords": chunk["keywords"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk in chunks
    ]

    return FAISS.from_documents(
        documents,
        embeddings,
    )


def search_events(
    index: FAISS,
    query: str,
    k: int = 5,
):
    """Recherche les événements les plus proches sémantiquement."""

    return index.similarity_search(
        query,
        k=k,
    )


def save_faiss_index(
    index: FAISS,
    output_dir: str = "data/faiss_index",
) -> None:
    """Sauvegarde l'index FAISS localement."""

    index.save_local(output_dir)


def load_faiss_index(
    input_dir: str = "data/faiss_index",
) -> FAISS:
    """Recharge un index FAISS généré localement."""

    embeddings = get_embeddings()

    return FAISS.load_local(
        input_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def report_progress(
    callback: ProgressCallback | None,
    progress: int,
    step: str,
    processed: int | None = None,
    total: int | None = None,
) -> None:
    """Transmet l'avancement du rebuild si un callback est fourni."""

    if callback is not None:
        callback(
            progress,
            step,
            processed,
            total,
        )


def main(
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Construit et sauvegarde l'index FAISS complet."""

    agenda_uid = 20272888

    report_progress(
        progress_callback,
        5,
        "Initialisation de la reconstruction",
    )

    report_progress(
        progress_callback,
        10,
        "Récupération des événements OpenAgenda",
    )

    events = fetch_all_events(agenda_uid)

    report_progress(
        progress_callback,
        25,
        "Conversion des événements",
        len(events),
        len(events),
    )

    df = events_to_dataframe(events)

    report_progress(
        progress_callback,
        35,
        "Filtrage des événements",
    )

    df = filter_events(df)

    report_progress(
        progress_callback,
        45,
        "Préparation des textes pour les embeddings",
        len(df),
        len(df),
    )

    df = add_embedding_text(df)

    report_progress(
        progress_callback,
        55,
        "Découpage des événements en chunks",
    )

    chunks = chunk_events(df)

    report_progress(
        progress_callback,
        65,
        "Génération des embeddings et construction FAISS",
        0,
        len(chunks),
    )

    index = build_faiss_index(chunks)

    report_progress(
        progress_callback,
        90,
        "Vérification de l'index FAISS",
        index.index.ntotal,
        len(chunks),
    )

    if index.index.ntotal != len(chunks):
        raise RuntimeError(
            f"FAISS index mismatch: "
            f"{index.index.ntotal} vectors "
            f"for {len(chunks)} chunks"
        )

    report_progress(
        progress_callback,
        95,
        "Sauvegarde de l'index FAISS",
        index.index.ntotal,
        len(chunks),
    )

    save_faiss_index(index)

    report_progress(
        progress_callback,
        100,
        "Reconstruction FAISS terminée",
        index.index.ntotal,
        len(chunks),
    )

    print(f"Événements : {len(df)}")
    print(f"Chunks : {len(chunks)}")
    print(f"Vecteurs FAISS : {index.index.ntotal}")
    print(
        "Index FAISS sauvegardé dans "
        "data/faiss_index"
    )


if __name__ == "__main__":
    main()