import os
import logging
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config.settings import RAG_EMBEDDING_MODEL, RAG_VECTOR_DB_PATH

logger = logging.getLogger(__name__)

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
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory
            )
        )
        
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
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=name,
                metadata={"description": "Seismic modeling and geophysics knowledge base"}
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
            
        # Generate document ID if not provided
        if doc_id is None:
            doc_id = f"doc_{len(self.collection.get()['ids']) + 1}"
        
        # Create embedding
        embedding = self.embedding_model.encode(text).tolist()
        
        # Add to collection
        self.collection.add(
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
            doc_ids = [f"doc_{len(self.collection.get()['ids']) + i + 1}" for i in range(len(texts))]
        
        # Create embeddings in batch for efficiency
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=doc_ids
        )
        
        logger.info(f"Added {len(texts)} documents to collection")
    
    def search(self, query: str, top_k: int = 5, domain: Optional[str] = None, 
               similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
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
                # Convert distance to similarity score (ChromaDB uses L2 distance)
                # Lower distance = higher similarity
                similarity_score = 1.0 / (1.0 + distance)
                
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
        
        logger.debug(f"Search query '{query}' returned {len(processed_results)} results")
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
        """Clear all documents from the collection."""
        self.collection.delete(where={})
        logger.info("Collection cleared")
    
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