from typing import Dict, List, Any, Optional
import os
import tempfile
from knowledge.vector_db import VectorDatabase
from knowledge.topics.ricker_wavelets import RICKER_KNOWLEDGE
from knowledge.topics.wedge_modeling import WEDGE_KNOWLEDGE
from knowledge.topics.seismic_properties import SEISMIC_PROPERTIES_KNOWLEDGE
from knowledge.topics.rock_physics import ROCK_PHYSICS_KNOWLEDGE

# Initialize the global vector database
_knowledge_db = None

def _get_knowledge_db():
    """
    Get or initialize the knowledge vector database with all topics.
    
    Returns:
        VectorDatabase: The initialized vector database
    """
    global _knowledge_db
    
    if _knowledge_db is None:
        _knowledge_db = VectorDatabase()
        
        # Add Ricker wavelet knowledge
        for subtopic, content in RICKER_KNOWLEDGE.items():
            if isinstance(content, str):
                _knowledge_db.add_document(
                    text=content,
                    metadata={'domain': 'ricker', 'topic': subtopic}
                )
        
        # Add wedge modeling knowledge
        for subtopic, content in WEDGE_KNOWLEDGE.items():
            if isinstance(content, str):
                _knowledge_db.add_document(
                    text=content,
                    metadata={'domain': 'wedge', 'topic': subtopic}
                )
        
        # Add seismic properties knowledge
        for subtopic, content in SEISMIC_PROPERTIES_KNOWLEDGE.items():
            if isinstance(content, str):
                _knowledge_db.add_document(
                    text=content,
                    metadata={'domain': 'seismic_properties', 'topic': subtopic}
                )
        
        # Add rock physics knowledge
        for subtopic, content in ROCK_PHYSICS_KNOWLEDGE.items():
            if isinstance(content, str):
                _knowledge_db.add_document(
                    text=content,
                    metadata={'domain': 'rock_physics', 'topic': subtopic}
                )
    
    return _knowledge_db


def knowledge_rag(query: str, domain: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
    """
    Retrieve knowledge using RAG (Retrieval-Augmented Generation) across all topics.
    
    Args:
        query: The user's query
        domain: Optional domain to restrict search (ricker, wedge, seismic_properties, rock_physics)
        top_k: Number of most relevant documents to retrieve
        
    Returns:
        Dict containing retrieved information and metadata
    """
    # Get the vector database
    db = _get_knowledge_db()
    
    # Search for relevant documents
    results = db.search(query, top_k=top_k, domain=domain)
    
    # Format the response
    formatted_results = []
    for i, result in enumerate(results):
        formatted_results.append({
            'content': result['document'],
            'domain': result['metadata'].get('domain', 'unknown'),
            'topic': result['metadata'].get('topic', 'unknown'),
            'relevance_score': float(result['score'])
        })
    
    # Create the response
    response = {
        'query': query,
        'domain': domain,
        'results': formatted_results,
        'total_results': len(formatted_results)
    }
    
    return response