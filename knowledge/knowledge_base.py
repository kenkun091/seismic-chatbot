from typing import Dict, Any, Optional
from .topics.ricker_wavelets import RICKER_KNOWLEDGE
from .topics.wedge_modeling import WEDGE_KNOWLEDGE
from .topics.seismic_properties import SEISMIC_PROPERTIES_KNOWLEDGE
from .topics.rock_physics import ROCK_PHYSICS_KNOWLEDGE
from .rag_system import RAGSystem

class KnowledgeBase:
    def __init__(self, llm_client=None):
        """Initialize the knowledge base with topic modules and RAG system.

        llm_client: shared, token/trace-accounted LLM client; RAGSystem builds
        its own when None.
        """
        self.topics = {
            'ricker': RICKER_KNOWLEDGE,
            'wedge': WEDGE_KNOWLEDGE,
            'seismic_properties': SEISMIC_PROPERTIES_KNOWLEDGE,
            'rock_physics': ROCK_PHYSICS_KNOWLEDGE
        }

        # Initialize RAG system
        self.rag_system = RAGSystem(llm_client=llm_client)
        
        # Populate the vector database with knowledge topics
        self._populate_vector_db()
    
    def _populate_vector_db(self):
        """Populate the vector database with all knowledge topics."""
        try:
            self.rag_system.populate_knowledge_base(self.topics)
        except Exception as e:
            print(f"Warning: Could not populate vector database: {e}")
            print("RAG functionality will be limited until database is populated.")
    
    def get_topic_response(self, topic: str, subtopic: Optional[str] = None) -> str:
        """
        Get a response for a specific topic and subtopic.
        
        Args:
            topic: The main topic to get information about
            subtopic: Optional specific aspect of the topic
            
        Returns:
            str: The knowledge base response
        """
        topic = topic.lower()
        
        # Get the topic knowledge
        topic_knowledge = self.topics.get(topic)
        if not topic_knowledge:
            return self._get_default_response()
            
        # If no subtopic specified, return topic overview
        if not subtopic:
            return topic_knowledge.get('overview', self._get_default_response())
            
        # Get specific subtopic response
        return topic_knowledge.get(subtopic, topic_knowledge.get('overview', self._get_default_response()))
    
    def query_knowledge(self, query: str, domain: Optional[str] = None,
                        context_manager: Any = None) -> Dict[str, Any]:
        """RAG query. context_manager (optional) receives token/trace
        accounting for the generation LLM call."""
        return self.rag_system.retrieve_and_generate(
            query, domain, context_manager=context_manager)
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """
        Get information about the current knowledge base.
        
        Returns:
            Dict with knowledge base statistics
        """
        return self.rag_system.get_knowledge_base_info()
    
    def _get_default_response(self) -> str:
        """
        Get the default response when no specific topic is found.
        
        Returns:
            str: Default response text
        """
        return """I can help answer questions about various **seismic modeling and geophysics** topics:

## 🌊 **Wavelets & Sources**
- Ricker wavelets: properties, frequency content, creation
- Source signatures: zero-phase vs minimum-phase
- Wavelet selection for different applications

## 📊 **Forward Modeling**
- Wedge models: thin bed effects, tuning phenomena
- Reflectivity computation: impedance contrasts
- Synthetic seismogram generation
- 1D, 2D, and 3D modeling approaches

## 🔍 **Seismic Properties**
- Frequency, bandwidth, and resolution relationships
- Velocity models and density effects
- Attenuation and quality factor (Q)
- Amplitude analysis and AVO

## 🎯 **Key Concepts**
- Tuning thickness and interference effects
- Seismic resolution limits and detection
- Wave propagation physics
- Reflection coefficients and polarity

## 🛠️ **Practical Applications**
- Survey design and acquisition planning
- Processing parameter optimization
- Interpretation workflows
- Reservoir characterization

**Example Questions:**
- *"What determines seismic resolution?"*
- *"How do I choose the right wavelet frequency?"*
- *"Explain tuning effects in thin beds"*
- *"What's the difference between zero-phase and minimum-phase wavelets?"*

**💡 Tip:** You can also ask me specific questions about any of these topics, and I'll use my knowledge base to provide detailed, accurate answers!"""