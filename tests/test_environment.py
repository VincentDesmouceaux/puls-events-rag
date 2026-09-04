def test_rag_dependencies_import():
    import faiss

    from langchain_community.vectorstores import FAISS
    from langchain_mistralai import (
        ChatMistralAI,
        MistralAIEmbeddings,
    )
    from mistralai.client import Mistral

    assert faiss is not None
    assert FAISS is not None
    assert ChatMistralAI is not None
    assert MistralAIEmbeddings is not None
    assert Mistral is not None