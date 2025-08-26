import re
import logging
from typing import List, Dict, Any
from config.settings import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles document processing including chunking and metadata extraction
    for optimal RAG performance.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size or RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or RAG_CHUNK_OVERLAP
        
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text: The text to chunk
            metadata: Metadata to associate with each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if metadata is None:
            metadata = {}
            
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Split into sentences first (better semantic boundaries)
        sentences = self._split_into_sentences(cleaned_text)
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self._create_chunk(current_chunk, metadata, chunk_id))
                chunk_id += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Take the last part of the previous chunk for overlap
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += sentence
                
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk, metadata, chunk_id))
            
        logger.info(f"Chunked text into {len(chunks)} chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for better chunking.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with chunking
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\{\}]', '', text)
        
        # Normalize line breaks
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Split on sentence endings (., !, ?) followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _create_chunk(self, text: str, metadata: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
        """
        Create a chunk dictionary with metadata.
        
        Args:
            text: Chunk text
            metadata: Base metadata
            chunk_id: Unique chunk identifier
            
        Returns:
            Chunk dictionary
        """
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'chunk_id': chunk_id,
            'chunk_size': len(text),
            'is_chunk': True
        })
        
        return {
            'text': text,
            'metadata': chunk_metadata
        }
    
    def process_knowledge_topics(self, knowledge_dict: Dict[str, Any], domain: str) -> List[Dict[str, Any]]:
        """
        Process knowledge topics into chunks for vector storage.
        
        Args:
            knowledge_dict: Dictionary of knowledge topics
            domain: Domain identifier for the knowledge
            
        Returns:
            List of processed chunks
        """
        all_chunks = []
        
        for topic, content in knowledge_dict.items():
            if isinstance(content, str):
                # Create metadata for this topic
                metadata = {
                    'domain': domain,
                    'topic': topic,
                    'content_type': 'knowledge_topic'
                }
                
                # Chunk the content
                chunks = self.chunk_text(content, metadata)
                all_chunks.extend(chunks)
                
        logger.info(f"Processed {domain} knowledge into {len(all_chunks)} chunks")
        return all_chunks
    
    def merge_chunks(self, chunks: List[Dict[str, Any]], max_length: int = None) -> str:
        """
        Merge chunks back into coherent text (useful for response generation).
        
        Args:
            chunks: List of chunk dictionaries
            max_length: Maximum length of merged text
            
        Returns:
            Merged text
        """
        if not chunks:
            return ""
            
        # Sort chunks by chunk_id to maintain order
        sorted_chunks = sorted(chunks, key=lambda x: x['metadata'].get('chunk_id', 0))
        
        # Merge text
        merged_text = " ".join([chunk['text'] for chunk in sorted_chunks])
        
        # Truncate if max_length specified
        if max_length and len(merged_text) > max_length:
            merged_text = merged_text[:max_length] + "..."
            
        return merged_text
