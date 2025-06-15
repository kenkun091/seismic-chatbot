from typing import Dict, Any, Optional
from .topics.ricker_wavelets import RICKER_KNOWLEDGE
from .topics.wedge_modeling import WEDGE_KNOWLEDGE
from .topics.seismic_properties import SEISMIC_PROPERTIES_KNOWLEDGE

class KnowledgeBase:
    def __init__(self):
        """Initialize the knowledge base with topic modules."""
        self.topics = {
            'ricker': RICKER_KNOWLEDGE,
            'wedge': WEDGE_KNOWLEDGE,
            'seismic_properties': SEISMIC_PROPERTIES_KNOWLEDGE
        }

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
- *"What's the difference between zero-phase and minimum-phase wavelets?"*""" 