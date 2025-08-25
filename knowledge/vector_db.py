from typing import Dict, List, Any, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorDatabase:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the vector database with an embedding model.
        
        Args:
            model_name: The name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """
        Add a document to the vector database.
        
        Args:
            text: The document text to embed
            metadata: Optional metadata associated with the document
        """
        if metadata is None:
            metadata = {}
            
        # Create embedding
        embedding = self.model.encode(text)
        
        # Store document and embedding
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
        
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Add multiple documents to the vector database.
        
        Args:
            texts: List of document texts to embed
            metadatas: Optional list of metadata dictionaries
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        # Create embeddings in batch for efficiency
        embeddings = self.model.encode(texts)
        
        # Store documents and embeddings
        self.documents.extend(texts)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadatas)
        
    def search(self, query: str, top_k: int = 3, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search the vector database for documents similar to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            domain: Optional domain filter to restrict search to specific knowledge domain
            
        Returns:
            List of dictionaries containing document, score, and metadata
        """
        # Create query embedding
        query_embedding = self.model.encode(query)
        
        # Calculate cosine similarity
        similarities = []
        domain_matches = []
        
        for i, doc_embedding in enumerate(self.embeddings):
            # Check domain filter if specified
            if domain is not None:
                doc_domain = self.metadata[i].get('domain')
                if doc_domain != domain:
                    domain_matches.append(False)
                    similarities.append(0.0)  # Add placeholder similarity
                    continue
            
            domain_matches.append(True)
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
            
        # Get top k results
        if not similarities or all(s == 0.0 for s in similarities):
            return []
        
        # Get indices of documents that match domain filter and sort by similarity
        valid_indices = [i for i, matches in enumerate(domain_matches) if matches]
        valid_similarities = [similarities[i] for i in valid_indices]
        
        if not valid_indices:
            return []
            
        # Sort valid indices by similarity
        sorted_indices = np.argsort(valid_similarities)[-min(top_k, len(valid_indices)):][::-1]
        top_indices = [valid_indices[i] for i in sorted_indices]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': similarities[idx],
                'metadata': self.metadata[idx]
            })
            
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))