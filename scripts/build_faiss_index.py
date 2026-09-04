import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document

from scripts.fetch_openagenda import (
    add_embedding_text,
    chunk_events,
    events_to_dataframe,
    fetch_all_events,
    filter_events,
)

load_dotenv()

def build_faiss_index(chunks: list[dict]) -> FAISS:
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "uid": chunk["uid"],
                "title": chunk["title"],
                "city": chunk["city"],
                "address": chunk["address"],
                "start_date": str(chunk["start_date"]),
                "end_date": str(chunk["end_date"]),
                "keywords": chunk["keywords"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk in chunks
    ]

    return FAISS.from_documents(documents, embeddings)
def search_events(
    index: FAISS,
    query: str,
    k: int = 5,
):
    return index.similarity_search(query, k=k)
def save_faiss_index(
    index: FAISS,
    output_dir: str = "data/faiss_index",
) -> None:
    index.save_local(output_dir)

def load_faiss_index(
    input_dir: str = "data/faiss_index",
) -> FAISS:
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )

    return FAISS.load_local(
        input_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
def main() -> None:
    agenda_uid = 20272888

    events = fetch_all_events(agenda_uid)
    df = events_to_dataframe(events)
    df = filter_events(df)
    df = add_embedding_text(df)
    chunks = chunk_events(df)

    index = build_faiss_index(chunks)

    if index.index.ntotal != len(chunks):
        raise RuntimeError(
            f"FAISS index mismatch: {index.index.ntotal} vectors "
            f"for {len(chunks)} chunks"
        )

    save_faiss_index(index)

    print(f"Événements : {len(df)}")
    print(f"Chunks : {len(chunks)}")
    print(f"Vecteurs FAISS : {index.index.ntotal}")
    print("Index FAISS sauvegardé dans data/faiss_index")


if __name__ == "__main__":
    main()
