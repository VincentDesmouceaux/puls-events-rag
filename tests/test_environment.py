def test_rag_dependencies_import():
    import faiss

    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from mistralai.client import Mistral

    assert faiss is not None
    assert FAISS is not None
    assert HuggingFaceEmbeddings is not None
    assert Mistral is not None