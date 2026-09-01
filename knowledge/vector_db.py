import os
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from config.settings import RAG_EMBEDDING_MODEL, RAG_VECTOR_DB_PATH

logger = logging.getLogger(__name__)


def content_id(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Deterministic ID derived from a document's content + metadata.

    Same content -> same ID across runs, so repeated population upserts in place
    instead of appending duplicates (the root cause of unbounded store growth).
    """
    h = hashlib.sha1()
    h.update((text or "").encode("utf-8"))
    h.update(json.dumps(metadata or {}, sort_keys=True).encode("utf-8"))
    return h.hexdigest()

class VectorDatabase:
    """
    A proper vector database implementation using ChromaDB for persistence
    and better performance than in-memory storage.
    """
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize the vector database with ChromaDB.
        
        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory or RAG_VECTOR_DB_PATH
        
        # Ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with new API
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create the collection
        self.collection = self._get_or_create_collection()
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(RAG_EMBEDDING_MODEL)
        
        logger.info(f"Vector database initialized at {self.persist_directory}")
        
    def _get_or_create_collection(self, name: str = "seismic_knowledge") -> chromadb.Collection:
        """
        Get existing collection or create a new one.
        
        Args:
            name: Name of the collection
            
        Returns:
            ChromaDB collection
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=name)
            logger.info(f"Using existing collection: {name}")
        except Exception:
            # Create new collection if it doesn't exist. Use COSINE space so the
            # distance->similarity mapping is interpretable (cosine = 1 - distance)
            # and the similarity threshold means a real cosine similarity.
            collection = self.client.create_collection(
                name=name,
                metadata={
                    "description": "Seismic modeling and geophysics knowledge base",
                    "hnsw:space": "cosine",
                }
            )
            logger.info(f"Created new collection: {name}")
        
        return collection
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None, doc_id: str = None):
        """
        Add a document to the vector database.
        
        Args:
            text: The document text to embed
            metadata: Optional metadata associated with the document
            doc_id: Optional document ID (auto-generated if not provided)
        """
        if metadata is None:
            metadata = {}

        # Deterministic, content-derived ID so re-population is idempotent.
        if doc_id is None:
            doc_id = content_id(text, metadata)

        # Skip if this exact content is already stored — avoids re-embedding cost
        # on every startup and prevents duplicate accumulation.
        existing = self.collection.get(ids=[doc_id])
        if existing and existing.get("ids"):
            return

        # Create embedding
        embedding = self.embedding_model.encode(text).tolist()

        # Upsert (idempotent) rather than add (which would duplicate on repeat).
        self.collection.upsert(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )

        logger.debug(f"Added document {doc_id} to collection")
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None, doc_ids: List[str] = None):
        """
        Add multiple documents to the vector database.
        
        Args:
            texts: List of document texts to embed
            metadatas: Optional list of metadata dictionaries
            doc_ids: Optional list of document IDs
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        if doc_ids is None:
            doc_ids = [content_id(t, m) for t, m in zip(texts, metadatas)]

        # Embed and upsert only the documents not already stored.
        existing = set(self.collection.get(ids=doc_ids).get("ids", []))
        new = [(t, m, i) for t, m, i in zip(texts, metadatas, doc_ids) if i not in existing]
        if not new:
            logger.info("All documents already present; nothing to add")
            return

        new_texts = [t for t, _, _ in new]
        new_metas = [m for _, m, _ in new]
        new_ids = [i for _, _, i in new]
        embeddings = self.embedding_model.encode(new_texts).tolist()

        self.collection.upsert(
            documents=new_texts,
            embeddings=embeddings,
            metadatas=new_metas,
            ids=new_ids
        )

        logger.info(f"Added {len(new_ids)} documents to collection")
    
    def search(self, query: str, top_k: int = 5, domain: Optional[str] = None,
               similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search the vector database for documents similar to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            domain: Optional domain filter to restrict search
            similarity_threshold: Minimum similarity score to include results
            
        Returns:
            List of dictionaries containing document, score, and metadata
        """
        # Create query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Prepare where clause for domain filtering
        where_clause = None
        if domain is not None:
            where_clause = {"domain": domain}
        
        # Search the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process and filter results
        processed_results = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                # Cosine space: distance = 1 - cosine_similarity, so similarity is
                # a true cosine similarity in [-1, 1] (typically [0, 1] here).
                similarity_score = 1.0 - distance

                # Filter by similarity threshold
                if similarity_score >= similarity_threshold:
                    processed_results.append({
                        'document': doc,
                        'score': similarity_score,
                        'metadata': metadata,
                        'distance': distance
                    })
        
        # Sort by similarity score (highest first)
        processed_results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Search query '{query}' returned {len(processed_results)} results")
        return processed_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection statistics
        """
        collection_data = self.collection.get()
        return {
            'name': self.collection.name,
            'count': len(collection_data['ids']),
            'metadata': self.collection.metadata
        }
    
    def clear_collection(self):
        """Clear all documents from the collection.

        Deletes by explicit IDs; ``delete(where={})`` is rejected as an empty
        filter by newer ChromaDB versions.
        """
        ids = self.collection.get().get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        logger.info(f"Collection cleared ({len(ids)} documents removed)")
    
    def delete_documents(self, doc_ids: List[str]):
        """
        Delete specific documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete
        """
        self.collection.delete(ids=doc_ids)
        logger.info(f"Deleted {len(doc_ids)} documents")
    
    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None):
        """
        Update an existing document.
        
        Args:
            doc_id: ID of the document to update
            text: New text content
            metadata: New metadata
        """
        # Delete the old document
        self.collection.delete(ids=[doc_id])
        
        # Add the updated document
        self.add_document(text, metadata, doc_id)
        
        logger.info(f"Updated document {doc_id}")