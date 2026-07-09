from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document as LIDocument
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient
from os import path


def _get_embed_model():
    """Build the embedding model from settings."""
    import json, os

    settings_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "settings.json")
    )
    try:
        with open(settings_path) as f:
            settings = json.load(f)
    except Exception:
        settings = {}

    emb = settings.get("embedding", {})
    emb_provider = emb.get("provider", "ollama")
    emb_model_name = emb.get("model", "nomic-embed-text:latest")

    if emb_provider == "ollama":
        providers = settings.get("providers", {})
        ollama_cfg = providers.get("ollama", {})
        base_url = ollama_cfg.get("base_url", "http://localhost:11434")
        return OllamaEmbedding(
            model_name=emb_model_name,
            base_url=base_url,
        )

    if emb_provider in ("openai", "alibaba"):
        from llama_index.embeddings.openai import OpenAIEmbedding
        providers = settings.get("providers", {})
        cfg = providers.get(emb_provider, {})
        api_key = cfg.get("api_key", "")
        base_url = cfg.get("base_url", "https://api.openai.com/v1")
        return OpenAIEmbedding(
            model_name=emb_model_name,
            api_key=api_key,
            api_base=base_url,
        )

    # Fallback: local Ollama
    return OllamaEmbedding(
        model_name="nomic-embed-text:latest",
        base_url="http://localhost:11434",
    )


# Initialize embedding model (lazy - built on first use via _get_embed_model)
embed_model = None

# Initialize ChromaDB client
chroma_client = PersistentClient(path="./.chroma_db")

current_index = None
current_collection_name = None


class IndexingCancelled(Exception):
    """Raised when an indexing job is cancelled mid-run."""


def _ensure_embed_model():
    global embed_model
    if embed_model is None:
        embed_model = _get_embed_model()
    return embed_model


def _extract_pdf_text(file_path: str) -> str:
    """Extract text from a PDF using pypdf (page by page)."""
    import pypdf

    reader = pypdf.PdfReader(file_path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts).strip()


def _read_documents(folder_path: str, check=None):
    """
    Yield llama-index Document objects from a folder, extracting real text:
      - .txt / .md  -> read as UTF-8
      - .pdf        -> pypdf text extraction (NOT raw bytes)

    Files that yield no extractable text (e.g. scanned/image-only PDFs) are
    skipped. Each yielded Document carries file_name/file_path/file_type
    metadata so the UI can list source files.
    """
    import os

    supported = {".txt", ".md", ".pdf"}

    for root, _dirs, files in os.walk(folder_path):
        for name in sorted(files):
            if check:
                check()
            ext = os.path.splitext(name)[1].lower()
            if ext not in supported:
                continue

            full = os.path.join(root, name)
            try:
                if ext == ".pdf":
                    text = _extract_pdf_text(full)
                    file_type = "application/pdf"
                else:
                    with open(full, "r", encoding="utf-8",
                              errors="ignore") as f:
                        text = f.read().strip()
                    file_type = "text/markdown" if ext == ".md" else "text/plain"
            except Exception:
                continue

            if not text:
                continue

            yield LIDocument(
                text=text,
                metadata={
                    "file_name": name,
                    "file_path": full,
                    "file_type": file_type,
                },
            )


def create_index_from_folder_cancellable(
    folder_path: str,
    collection_name: str = "default",
    is_cancelled=None,
    on_progress=None,
    on_phase=None,
):
    """
    Cancellable variant of create_index_from_folder.

    Parses and embeds documents one at a time, checking `is_cancelled()`
    frequently (before reading, before parsing each file, and before each
    node embedding) so a running job can be stopped quickly at any phase.
    On cancel, the partially-built collection is deleted and
    IndexingCancelled is raised.

    Args:
        is_cancelled: callable -> bool, polled often.
        on_progress: callable(done: int, total: int) for embedded nodes.
        on_phase: callable(phase: str) — "reading" | "embedding".
    """
    global current_index, current_collection_name

    from llama_index.core.node_parser import SentenceSplitter

    embed = _ensure_embed_model()

    def cancelled() -> bool:
        return bool(is_cancelled and is_cancelled())

    def check():
        if cancelled():
            raise IndexingCancelled()

    if not path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")

    check()
    if on_phase:
        on_phase("reading")

    splitter = SentenceSplitter()

    all_nodes = []
    docs_indexed = 0
    for doc in _read_documents(folder_path, check):
        check()
        all_nodes.extend(splitter.get_nodes_from_documents([doc]))
        docs_indexed += 1

    if docs_indexed == 0:
        raise ValueError(
            f"No readable text found in {folder_path}. "
            "Supported: .txt, .md, and text-based .pdf files."
        )

    total = len(all_nodes)
    if on_progress:
        on_progress(0, total)

    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    chroma_collection = chroma_client.create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
        embed_model=embed,
    )

    if on_phase:
        on_phase("embedding")

    BATCH = 16
    nodes_done = 0
    try:
        for start in range(0, total, BATCH):
            check()
            batch = all_nodes[start:start + BATCH]
            index.insert_nodes(batch)
            nodes_done += len(batch)
            if on_progress:
                on_progress(nodes_done, total)
    except IndexingCancelled:
        try:
            chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass
        raise

    current_index = index
    current_collection_name = collection_name

    return {
        "status": "success",
        "documents_indexed": docs_indexed,
        "nodes_indexed": nodes_done,
        "collection": collection_name,
    }


def load_existing_index(collection_name: str = "default"):
    """
    Load an existing index from ChromaDB.
    """
    global current_index, current_collection_name

    embed = _ensure_embed_model()

    try:
        chroma_collection = chroma_client.get_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        current_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed,
        )

        current_collection_name = collection_name
        return {"status": "success", "collection": collection_name}
    except Exception as e:
        raise ValueError(f"Failed to load collection '{collection_name}': {str(e)}")


def get_context_for_llm(question: str, top_k: int = 3):
    """
    Get relevant context to inject into LLM prompt.
    Returns tuple: (context_text, sources_list)
    """
    if current_index is None:
        return "", []

    retriever = current_index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    context_parts = []
    sources = []

    for i, node in enumerate(nodes, 1):
        context_parts.append(f"[Source {i}]\n{node.node.text}\n")
        meta = node.node.metadata or {}
        sources.append({
            "id": i,
            "title": meta.get("file_name")
            or (
                meta.get("file_path", "").rsplit("/", 1)[-1]
                if meta.get("file_path") else "Document"
            ),
            "file_path": meta.get("file_path"),
            "text": node.node.text[:200] + "...",
            "score": node.score,
            "metadata": meta,
        })

    return "\n".join(context_parts), sources


def list_collections():
    collections = chroma_client.list_collections()
    return [{"name": col.name, "count": col.count()} for col in collections]


def get_current_collection_info():
    if current_index is None:
        return {"status": "No collection loaded"}

    return {
        "collection_name": current_collection_name,
        "status": "loaded"
    }


def delete_collection(collection_name: str):
    global current_index, current_collection_name

    try:
        existing = {c.name for c in chroma_client.list_collections()}
        if collection_name not in existing:
            raise ValueError(f"Collection '{collection_name}' not found")
        chroma_client.delete_collection(name=collection_name)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to delete collection '{collection_name}': {str(e)}"
        )

    if current_collection_name == collection_name:
        current_index = None
        current_collection_name = None

    return {"status": "deleted", "collection": collection_name}


def get_collection_files(collection_name: str):
    try:
        existing = {c.name for c in chroma_client.list_collections()}
        if collection_name not in existing:
            raise ValueError(f"Collection '{collection_name}' not found")
        collection = chroma_client.get_collection(name=collection_name)
        data = collection.get(include=["metadatas"])
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to read collection '{collection_name}': {str(e)}"
        )

    seen = {}
    for md in data.get("metadatas") or []:
        if not md:
            continue
        path_key = md.get("file_path") or md.get("file_name")
        if not path_key or path_key in seen:
            continue
        seen[path_key] = {
            "file_name": md.get("file_name")
            or (path_key.rsplit("/", 1)[-1] if path_key else "unknown"),
            "file_path": md.get("file_path"),
            "file_type": md.get("file_type"),
        }

    files = list(seen.values())
    files.sort(key=lambda f: (f.get("file_name") or "").lower())
    return {"collection": collection_name, "files": files}


def get_collection_source_folder(collection_name: str):
    import os

    info = get_collection_files(collection_name)
    paths = [
        f["file_path"] for f in info["files"] if f.get("file_path")
    ]
    if not paths:
        return {"collection": collection_name, "folder_path": None}

    if len(paths) == 1:
        folder = os.path.dirname(paths[0])
    else:
        folder = os.path.commonpath(paths)
        if not os.path.isdir(folder):
            folder = os.path.dirname(folder)

    return {"collection": collection_name, "folder_path": folder}
