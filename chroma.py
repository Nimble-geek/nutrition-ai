import os
import chromadb
from chromadb.utils import embedding_functions


class ChromaVectorStore:
    """
    Lightweight ChromaDB wrapper optimized for Render deployments
    """

    def __init__(self, db_path: str = None, collection_name: str = "diet_col"):
        """
        Initialize ChromaDB client and embedding function
        """

        # Use environment variable for flexibility in Render
        self.db_path = db_path or os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.collection_name = collection_name

        # Ensure DB directory exists
        os.makedirs(self.db_path, exist_ok=True)

        # Lightweight embedding model
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Persistent client
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Load or create collection
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """
        Get existing collection or create a new one
        """
        try:
            return self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except Exception:
            return self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )

    def prepare_documents(self, splits):
        """
        Extract documents, ids, and metadata
        """

        documents = []
        ids = []
        metadatas = []

        for doc in splits:

            documents.append(doc.page_content)

            filename = doc.metadata["dl_meta"]["origin"]["filename"]
            ref = doc.metadata["dl_meta"]["doc_items"][0]["self_ref"]
            page = doc.metadata["dl_meta"]["doc_items"][0]["prov"][0]["page_no"]

            ids.append(f"{filename}_{ref}")

            metadatas.append({
                "source": filename,
                "page": page
            })

        return documents, metadatas, ids

    def upsert_documents(self, splits, batch_size: int = 100):
        """
        Insert documents in batches (better for Render memory limits)
        """

        documents, metadatas, ids = self.prepare_documents(splits)

        for i in range(0, len(documents), batch_size):

            self.collection.upsert(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )

    def query(self, query_text: str, n_results: int = 3):
        """
        Query the vector database
        """

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        return results