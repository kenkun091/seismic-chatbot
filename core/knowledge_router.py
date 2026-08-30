"""Intent split + knowledge (RAG) path, shared by both chat modes."""
import json
import logging

from core.turn_trace import emit_event

logger = logging.getLogger(__name__)


class KnowledgeRouter:
    def __init__(self, llm_client, knowledge_base, context_manager=None):
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base
        self.context_manager = context_manager

    def _simple(self, system_prompt: str, user_prompt: str) -> str:
        """get_simple_completion with token/trace accounting; tolerates legacy
        fakes whose signature lacks the context_manager kwarg."""
        try:
            return self.llm_client.get_simple_completion(
                system_prompt, user_prompt, context_manager=self.context_manager)
        except TypeError:
            return self.llm_client.get_simple_completion(system_prompt, user_prompt)

    def classify(self, user_input: str) -> dict:
        """Three-way intent decision with provenance: which branch decided."""
        if user_input.lstrip().startswith("[image attached"):
            verdict = {"is_knowledge": False, "via": "image_shortcut"}
        else:
            try:
                verdict = {"is_knowledge": self._classify_intent_with_llm(user_input),
                           "via": "llm"}
            except Exception as e:
                logger.error(f"LLM intent classification failed: {e}")
                verdict = {"is_knowledge": self._is_knowledge_question_keywords(user_input),
                           "via": "keyword_fallback"}
        label = "KNOWLEDGE" if verdict["is_knowledge"] else "TOOL"
        logger.info(f"intent: {label} (via {verdict['via']})")
        emit_event(self.context_manager, "intent", verdict=label, via=verdict["via"])
        return verdict

    def is_knowledge_question(self, user_input: str) -> bool:
        return self.classify(user_input)["is_knowledge"]

    def _classify_intent_with_llm(self, user_input: str) -> bool:
        """
        Use LLM to classify whether the input is a knowledge question.

        Args:
            user_input: The user's input text

        Returns:
            bool: True if this is a knowledge question that should use RAG
        """
        system_prompt = """You are an expert at classifying user intents in a seismic modeling chatbot context.

Your task is to determine if a user's input is a KNOWLEDGE QUESTION that should be answered using the knowledge base (RAG), or if it's a TOOL REQUEST that should use specific seismic modeling tools.

KNOWLEDGE QUESTIONS include:
- Questions asking for explanations, definitions, or descriptions
- Questions about concepts, principles, or theory
- Questions about relationships, effects, or trade-offs
- Questions starting with "what", "how", "why", "explain", "describe", "tell me about"
- Questions about seismic properties, resolution, frequency effects, etc.
- Questions seeking educational information or understanding

TOOL REQUESTS include:
- Requests to create, generate, or make something (e.g., "create a Ricker wavelet")
- Requests to plot, visualize, or display something
- Requests to calculate or compute specific values
- Requests to model or simulate something
- Requests with specific parameters or values

Examples:
- "How does frequency affect seismic resolution?" → KNOWLEDGE QUESTION
- "What is a Ricker wavelet?" → KNOWLEDGE QUESTION
- "Create a 30 Hz Ricker wavelet" → TOOL REQUEST
- "Plot the wedge model" → TOOL REQUEST
- "What are the trade-offs of higher frequency?" → KNOWLEDGE QUESTION
- "Make a wedge model with 100m thickness" → TOOL REQUEST

Respond with ONLY "KNOWLEDGE" or "TOOL" - no other text."""

        try:
            response = self._simple(system_prompt, user_input)
            response = response.strip().upper()

            # Log the classification for debugging
            logger.debug(f"LLM classified '{user_input[:50]}...' as: {response}")

            return response == "KNOWLEDGE"

        except Exception as e:
            logger.error(f"Error in LLM intent classification: {e}")
            raise e

    def classify_intent_detailed(self, user_input: str) -> dict:
        """
        Use LLM to classify user intent with detailed information.

        Args:
            user_input: The user's input text

        Returns:
            Dict with intent classification and confidence
        """
        system_prompt = """You are an expert at classifying user intents in a seismic modeling chatbot context.

Classify the user's intent and provide detailed information about it.

INTENT TYPES:
1. KNOWLEDGE_QUESTION - Questions seeking explanations, definitions, or educational information
2. TOOL_REQUEST - Requests to create, plot, calculate, or model something
3. MIXED - Both knowledge and tool components
4. UNCLEAR - Ambiguous or unclear intent

For each intent, also determine:
- CONFIDENCE: How confident you are (0.0 to 1.0)
- REASONING: Brief explanation of your classification
- SUGGESTED_ACTION: What the chatbot should do

Respond in JSON format:
{
    "intent": "KNOWLEDGE_QUESTION|TOOL_REQUEST|MIXED|UNCLEAR",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "suggested_action": "Use RAG|Use Tools|Ask for clarification|Both"
}"""

        try:
            response = self.llm_client.get_simple_completion(system_prompt, user_input)

            # Try to parse JSON response
            import json
            try:
                result = json.loads(response)
                logger.debug(f"Detailed classification: {result}")
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON response: {response}")
                return {
                    "intent": "UNCLEAR",
                    "confidence": 0.5,
                    "reasoning": "Failed to parse LLM response",
                    "suggested_action": "Ask for clarification"
                }

        except Exception as e:
            logger.error(f"Error in detailed intent classification: {e}")
            return {
                "intent": "UNCLEAR",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "suggested_action": "Use fallback classification"
            }

    def _is_knowledge_question_keywords(self, user_input: str) -> bool:
        """
        Fallback keyword-based detection for knowledge questions.

        Args:
            user_input: The user's input text

        Returns:
            bool: True if this should use RAG
        """
        # Keywords that indicate knowledge questions
        knowledge_keywords = [
            'what is', 'what are', 'explain', 'describe', 'how does', 'why does',
            'tell me about', 'what determines', 'what affects', 'what causes',
            'difference between', 'relationship between', 'definition of',
            'characteristics of', 'properties of', 'applications of',
            'how can', 'what happens', 'what is the', 'what are the',
            'can you explain', 'can you describe', 'what do you know',
            'trade-offs', 'advantages', 'disadvantages', 'benefits',
            'limitations', 'constraints', 'factors', 'influence',
            'impact', 'effect', 'role', 'significance', 'importance'
        ]

        user_input_lower = user_input.lower()

        # Check if input contains knowledge question patterns
        for keyword in knowledge_keywords:
            if keyword in user_input_lower:
                return True

        # Check if it's a question (ends with ?)
        if user_input.strip().endswith('?'):
            return True

        # Check if it's asking for explanation
        if any(word in user_input_lower for word in ['explain', 'describe', 'tell me']):
            return True

        # Check for seismic/geophysical concept questions
        seismic_concepts = [
            'frequency', 'resolution', 'bandwidth', 'wavelength', 'tuning',
            'impedance', 'velocity', 'density', 'attenuation', 'quality factor',
            'reflection', 'refraction', 'wavelet', 'ricker', 'wedge model',
            'seismic', 'geophysical', 'geology', 'petroleum', 'reservoir'
        ]

        # If it contains seismic concepts and is asking for information, use RAG
        if any(concept in user_input_lower for concept in seismic_concepts):
            if any(word in user_input_lower for word in ['what', 'how', 'why', 'explain', 'describe', 'tell']):
                return True

        return False

    def handle_knowledge_question(self, user_input: str) -> str:
        """
        Handle knowledge questions using RAG.

        Args:
            user_input: The user's question

        Returns:
            str: Generated response using RAG
        """
        try:
            # Use the knowledge base's RAG system
            try:
                rag_response = self.knowledge_base.query_knowledge(
                    user_input, context_manager=self.context_manager)
            except TypeError:
                rag_response = self.knowledge_base.query_knowledge(user_input)

            docs = rag_response.get('retrieved_documents') or []
            emit_event(self.context_manager, "rag",
                       rag_type=rag_response.get('rag_type'),
                       retrieved=rag_response.get('total_retrieved', 0),
                       scores=[round(d.get('score', 0.0), 4) for d in docs
                               if isinstance(d, dict)])

            if rag_response.get('rag_type') == 'retrieve_and_generate':
                # Successfully generated response
                response = rag_response['generated_response']

                # Add metadata about the retrieval
                retrieved_count = rag_response.get('total_retrieved', 0)
                if retrieved_count > 0:
                    response += f"\n\n*Based on {retrieved_count} relevant documents from the knowledge base.*"

                return response

            elif rag_response.get('rag_type') == 'no_results':
                # No relevant documents found - use LLM with general knowledge
                logger.info("No RAG results found, using LLM with general seismic knowledge")
                return self._handle_no_rag_results(user_input)

            else:
                # Error or fallback
                response = rag_response['generated_response']
                # Ensure we never return boolean values
                if isinstance(response, bool):
                    response = str(response)
                return response

        except Exception as e:
            logger.error(f"Error in RAG processing: {e}")
            # Fallback to LLM with general knowledge
            return self._handle_no_rag_results(user_input)

    def _handle_no_rag_results(self, user_input: str) -> str:
        """
        Handle cases when RAG doesn't find relevant documents by using LLM with general seismic knowledge.

        Args:
            user_input: The user's question

        Returns:
            str: LLM-generated response using general knowledge
        """
        try:
            # Create a comprehensive system prompt for seismic knowledge
            system_prompt = """You are an expert seismic modeling and geophysics assistant with extensive knowledge of:

**Core Seismic Concepts:**
- Wave propagation physics and properties
- Frequency, bandwidth, and resolution relationships
- Velocity, density, and impedance effects
- Reflection and refraction phenomena
- Attenuation and quality factor (Q)

**Wavelet Theory:**
- Ricker wavelets and their frequency characteristics
- Zero-phase vs minimum-phase wavelets
- Bandwidth and temporal resolution trade-offs
- Source signature design principles

**Forward Modeling:**
- Wedge models and tuning effects
- Thin bed analysis and resolution limits
- Synthetic seismogram generation
- AVO (Amplitude vs Offset) analysis

**Seismic Resolution:**
- Frequency vs resolution relationships
- Tuning thickness and interference effects
- Detection vs resolution limits
- Trade-offs between penetration and resolution

**Rock Physics:**
- Velocity-density relationships
- Fluid effects on seismic properties
- Porosity and permeability impacts
- Lithology identification methods

**Practical Applications:**
- Survey design and acquisition planning
- Processing parameter optimization
- Interpretation workflows
- Reservoir characterization

IMPORTANT — this question was NOT matched to the curated knowledge base, so you are
answering from general knowledge only. To avoid misleading the user:
1. Do NOT fabricate or invent specific numeric constants, coefficients, equations, or
   citations. If you are not confident in an exact value, say so rather than making one up.
2. Prefer qualitative explanations and clearly-labelled typical ranges over precise numbers.
3. Explicitly flag uncertainty and recommend authoritative references where appropriate.
4. Provide accurate, educational explanations; structure them logically.

Answer the user's question using your general knowledge of seismic modeling and geophysics,
within the constraints above."""

            # Generate response using the LLM
            response = self._simple(system_prompt, user_input)

            # Clearly label the answer as NOT grounded in the curated knowledge base.
            disclaimer = (
                "\n\n*⚠️ Not from the curated knowledge base — this is a general-knowledge "
                "answer and may contain inaccuracies. Verify specific values against an "
                "authoritative reference.*"
            )
            return (response + disclaimer).strip()

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            # Final fallback to basic knowledge base
            return self._fallback_knowledge_response(user_input)

    def _fallback_knowledge_response(self, user_input: str) -> str:
        """
        Fallback response when RAG fails.

        Args:
            user_input: The user's question

        Returns:
            str: Fallback response
        """
        # Try to extract topic from the question
        user_input_lower = user_input.lower()

        # Check for specific topics
        if any(word in user_input_lower for word in ['ricker', 'wavelet']):
            response = self.knowledge_base.get_topic_response('ricker', 'overview')
            # Ensure we never return boolean values
            if isinstance(response, bool):
                response = str(response)
            return response
        elif any(word in user_input_lower for word in ['wedge', 'model']):
            response = self.knowledge_base.get_topic_response('wedge', 'overview')
            # Ensure we never return boolean values
            if isinstance(response, bool):
                response = str(response)
            return response
        elif any(word in user_input_lower for word in ['seismic', 'resolution', 'frequency']):
            response = self.knowledge_base.get_topic_response('seismic_properties', 'overview')
            # Ensure we never return boolean values
            if isinstance(response, bool):
                response = str(response)
            return response
        elif any(word in user_input_lower for word in ['rock', 'physics', 'porosity', 'velocity']):
            response = self.knowledge_base.get_topic_response('rock_physics', 'overview')
            # Ensure we never return boolean values
            if isinstance(response, bool):
                response = str(response)
            return response
        else:
            response = self.knowledge_base.get_topic_response('ricker', 'overview')  # Default topic
            # Ensure we never return boolean values
            if isinstance(response, bool):
                response = str(response)
            return response
