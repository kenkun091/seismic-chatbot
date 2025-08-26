import logging
from typing import Dict, List, Any, Optional
from .vector_db import VectorDatabase
from .document_processor import DocumentProcessor
from core.llm_client import LLMClient
from config.settings import RAG_TOP_K, RAG_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

class RAGSystem:
    """
    A proper RAG (Retrieval-Augmented Generation) system that integrates
    document retrieval with LLM-based response generation.
    """
    
    def __init__(self, vector_db: VectorDatabase = None, llm_client: LLMClient = None):
        """
        Initialize the RAG system.
        
        Args:
            vector_db: Vector database instance
            llm_client: LLM client for generation
        """
        self.vector_db = vector_db or VectorDatabase()
        self.document_processor = DocumentProcessor()
        self.llm_client = llm_client or LLMClient()
        
        logger.info("RAG system initialized")
    
    def retrieve_and_generate(self, query: str, domain: Optional[str] = None, 
                            top_k: int = None, similarity_threshold: float = None) -> Dict[str, Any]:
        """
        Main RAG function: retrieve relevant documents and generate a response.
        
        Args:
            query: User's question
            domain: Optional domain to restrict search
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            Dictionary with generated response and metadata
        """
        # Set defaults
        top_k = top_k or RAG_TOP_K
        similarity_threshold = similarity_threshold or RAG_SIMILARITY_THRESHOLD
        
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(
                query, top_k, domain, similarity_threshold
            )
            
            if not retrieved_docs:
                return self._handle_no_results(query, domain)
            
            # Step 2: Generate response using retrieved context
            response = self._generate_response(query, retrieved_docs)
            
            # Step 3: Format and return results
            return {
                'query': query,
                'domain': domain,
                'generated_response': response,
                'retrieved_documents': retrieved_docs,
                'total_retrieved': len(retrieved_docs),
                'rag_type': 'retrieve_and_generate'
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return self._handle_error(query, str(e))
    
    def _retrieve_documents(self, query: str, top_k: int, domain: Optional[str], 
                           similarity_threshold: float) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector database.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            domain: Domain filter
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieved documents
        """
        logger.debug(f"Retrieving documents for query: {query}")
        
        # Search the vector database
        results = self.vector_db.search(
            query=query,
            top_k=top_k,
            domain=domain,
            similarity_threshold=similarity_threshold
        )
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def _generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the LLM and retrieved context.
        
        Args:
            query: User's question
            retrieved_docs: Retrieved documents
            
        Returns:
            Generated response
        """
        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)
        
        # Create the prompt for the LLM
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(query, context)
        
        try:
            # Generate response using the LLM
            response = self.llm_client.get_simple_completion(system_prompt, user_prompt)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback: return a summary of retrieved content
            return self._create_fallback_response(query, retrieved_docs)
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Prepare retrieved documents into a coherent context for the LLM.
        
        Args:
            retrieved_docs: Retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs):
            # Add document header
            domain = doc['metadata'].get('domain', 'unknown')
            topic = doc['metadata'].get('topic', 'unknown')
            score = doc['score']
            
            header = f"Document {i+1} (Domain: {domain}, Topic: {topic}, Relevance: {score:.3f}):"
            
            # Add document content
            content = doc['document']
            
            # Truncate very long content
            if len(content) > 500:
                content = content[:500] + "..."
            
            context_parts.append(f"{header}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for the LLM.
        
        Returns:
            System prompt string
        """
        return """You are an expert seismic modeling and geophysics assistant. Your role is to answer questions based on the provided context from a knowledge base.

Guidelines:
1. Answer questions using ONLY the information provided in the context
2. If the context doesn't contain enough information to answer the question, say so
3. Be accurate, concise, and educational
4. Use the context to provide specific, relevant information
5. If asked about seismic concepts, explain them clearly
6. Cite the specific documents you're using from the context

Format your response in a clear, structured way that directly addresses the user's question."""
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """
        Create the user prompt with context for the LLM.
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            User prompt string
        """
        return f"""Based on the following context, please answer this question:

Question: {query}

Context:
{context}

Please provide a comprehensive answer based on the context above."""
    
    def _handle_no_results(self, query: str, domain: Optional[str]) -> Dict[str, Any]:
        """
        Handle cases where no relevant documents are found.
        
        Args:
            query: User's question
            domain: Domain filter
            
        Returns:
            Response for no results
        """
        return {
            'query': query,
            'domain': domain,
            'generated_response': f"I couldn't find specific information about '{query}' in the knowledge base. This might be because:\n\n1. The topic isn't covered in our current knowledge base\n2. The question uses different terminology than what's in our documents\n3. The information might be in a different domain\n\nPlease try rephrasing your question or ask about a different seismic modeling topic.",
            'retrieved_documents': [],
            'total_retrieved': 0,
            'rag_type': 'no_results'
        }
    
    def _handle_error(self, query: str, error_message: str) -> Dict[str, Any]:
        """
        Handle errors in the RAG pipeline.
        
        Args:
            query: User's question
            error_message: Error description
            
        Returns:
            Error response
        """
        return {
            'query': query,
            'error': error_message,
            'generated_response': f"I encountered an error while processing your question: {error_message}. Please try again or rephrase your question.",
            'retrieved_documents': [],
            'total_retrieved': 0,
            'rag_type': 'error'
        }
    
    def _create_fallback_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Create a fallback response when LLM generation fails.
        
        Args:
            query: User's question
            retrieved_docs: Retrieved documents
            
        Returns:
            Fallback response
        """
        if not retrieved_docs:
            return f"I found some information about '{query}' but couldn't generate a comprehensive response. Please try rephrasing your question."
        
        # Create a simple summary of retrieved content
        summary_parts = [f"Based on the retrieved information about '{query}':"]
        
        for i, doc in enumerate(retrieved_docs[:3]):  # Limit to first 3 docs
            domain = doc['metadata'].get('domain', 'unknown')
            topic = doc['metadata'].get('topic', 'unknown')
            content = doc['document'][:200] + "..." if len(doc['document']) > 200 else doc['document']
            
            summary_parts.append(f"\nDocument {i+1} ({domain}/{topic}): {content}")
        
        return "\n".join(summary_parts)
    
    def populate_knowledge_base(self, knowledge_topics: Dict[str, Dict[str, Any]]):
        """
        Populate the vector database with knowledge topics.
        
        Args:
            knowledge_topics: Dictionary of knowledge domains and topics
        """
        logger.info("Populating knowledge base with topics...")
        
        total_chunks = 0
        
        for domain, topics in knowledge_topics.items():
            # Process topics into chunks
            chunks = self.document_processor.process_knowledge_topics(topics, domain)
            
            # Add chunks to vector database
            for chunk in chunks:
                self.vector_db.add_document(
                    text=chunk['text'],
                    metadata=chunk['metadata']
                )
            
            total_chunks += len(chunks)
            logger.info(f"Added {len(chunks)} chunks for domain: {domain}")
        
        logger.info(f"Knowledge base populated with {total_chunks} total chunks")
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """
        Get information about the current knowledge base.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        return self.vector_db.get_collection_info()
