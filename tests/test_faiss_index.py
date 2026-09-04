from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]


def test_faiss_indexes_all_documents():
    documents = [
        Document(page_content="Jazz à Paris", metadata={"uid": 1}),
        Document(page_content="Concert groove", metadata={"uid": 2}),
        Document(page_content="Jam session", metadata={"uid": 3}),
    ]

    index = FAISS.from_documents(documents, FakeEmbeddings())

    assert index.index.ntotal == len(documents)

def test_faiss_similarity_search_returns_documents():
    documents = [
        Document(page_content="Jazz à Paris", metadata={"uid": 1}),
        Document(page_content="Concert groove", metadata={"uid": 2}),
        Document(page_content="Jam session", metadata={"uid": 3}),
    ]

    index = FAISS.from_documents(documents, FakeEmbeddings())

    results = index.similarity_search("Jazz à Paris", k=2)

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)

def test_faiss_save_and_load(tmp_path):
    documents = [
        Document(page_content="Jazz à Paris", metadata={"uid": 1}),
        Document(page_content="Jam session", metadata={"uid": 2}),
    ]

    embeddings = FakeEmbeddings()
    index = FAISS.from_documents(documents, embeddings)

    index.save_local(str(tmp_path))

    loaded = FAISS.load_local(
        str(tmp_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    assert loaded.index.ntotal == 2