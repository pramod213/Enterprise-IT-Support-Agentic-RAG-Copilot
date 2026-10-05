import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.core.config import get_settings


settings = get_settings()

_embeddings = None
_vectorstore = None


# Embedding dimensions for supported Hugging Face models.
EMBEDDING_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    """Return the vector dimension for the configured embedding model."""
    name = (model_name or settings.embedding_model or "").strip()

    if not name:
        raise RuntimeError("EMBEDDING_MODEL is not configured")

    # Exact match.
    if name in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[name]

    # Case-insensitive match.
    normalized = name.lower()

    for model, dimension in EMBEDDING_DIMENSIONS.items():
        if model.lower() == normalized:
            return dimension

    # Pattern-based fallback.
    if "bge-small-en-v1.5" in normalized:
        return 384

    if "bge-base-en-v1.5" in normalized:
        return 768

    if "all-minilm" in normalized:
        return 384

    if "all-mpnet" in normalized:
        return 768

    raise ValueError(
        f"Unsupported embedding model '{name}' for Pinecone. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create and cache the Hugging Face embedding model."""
    global _embeddings

    if _embeddings is None:
        model_name = (settings.embedding_model or "").strip()

        if not model_name:
            raise RuntimeError("EMBEDDING_MODEL is missing")

        _embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

    return _embeddings


def ensure_index():
    """Create the Pinecone index if necessary and ensure its dimension matches."""
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(
        api_key=settings.pinecone_api_key,
    )

    index_name = settings.pinecone_index_name

    existing_indexes = pc.list_indexes()
    names = [item["name"] for item in existing_indexes]

    # Check whether the configured index already exists.
    if index_name in names:
        index_info = pc.describe_index(index_name)

        current_dimension = getattr(
            index_info,
            "dimension",
            None,
        )

        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        # Recreate the index when its dimension does not match
        # the configured Hugging Face embedding model.
        if (
            current_dimension is not None
            and current_dimension != desired_dimension
        ):
            print(
                f"Pinecone dimension mismatch: "
                f"existing={current_dimension}, "
                f"required={desired_dimension}. "
                "Deleting and recreating the index."
            )

            pc.delete_index(name=index_name)

            while index_name in [
                item["name"] for item in pc.list_indexes()
            ]:
                time.sleep(1)

    # Create the index when it doesn't exist.
    if index_name not in [
        item["name"] for item in pc.list_indexes()
    ]:
        pc.create_index(
            name=index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        # Wait until Pinecone reports the index as ready.
        while True:
            index_info = pc.describe_index(index_name)

            if index_info.status["ready"]:
                break

            time.sleep(1)

    return pc.Index(index_name)


def get_vectorstore() -> PineconeVectorStore:
    """Create and cache the Pinecone vector store."""
    global _vectorstore

    if _vectorstore is None:
        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():
    """Return the configured Pinecone retriever."""
    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": settings.top_k,
        }
    )


def add_documents(chunks):
    """Add document chunks to the Pinecone vector store."""
    store = get_vectorstore()

    return store.add_documents(chunks)