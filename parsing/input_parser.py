import re
import json
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class InputParser:
    def __init__(self):
        """Initialize the input parser."""
        self.question_keywords = {
            'what': 3, 'why': 3, 'how': 3, 'when': 2, 'where': 2, 'which': 2, 'who': 2,
            'explain': 3, 'tell me': 2, 'describe': 2, 'definition': 3, 'meaning': 2,
            'difference': 2, 'compare': 2, 'help': 1, 'understand': 2, 'learn': 2
        }
        
        self.action_keywords = {
            'create': 3, 'make': 3, 'generate': 3, 'build': 3, 'compute': 2, 'calculate': 2,
            'plot': 3, 'show': 2, 'display': 2, 'visualize': 2, 'model': 2, 'simulate': 2
        }

    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from LLM response.
        
        Args:
            response: The LLM response text
            
        Returns:
            Dict[str, Any]: Parsed JSON content
        """
        try:
            # Look for JSON block in the response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif response.startswith("{"):
                json_str = response
            else:
                # Try to find JSON-like content
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start != -1 and json_end != 0:
                    json_str = response[json_start:json_end]
                else:
                    raise ValueError("No JSON found in response")
            
            return json.loads(json_str)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing error: {e}")
            raise

    def classify_user_intent(self, user_input: str) -> str:
        """
        Classify user intent from input text.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: 'question', 'action', or 'unclear'
        """
        text = user_input.lower().strip()
        
        question_score = 0
        action_score = 0
        
        # Punctuation scoring
        if text.endswith('?'):
            question_score += 3
        
        if text.startswith(('what', 'why', 'how', 'when', 'where', 'which', 'who')):
            question_score += 3
        
        # Keyword scoring
        for keyword, weight in self.question_keywords.items():
            if keyword in text:
                question_score += weight
        
        for keyword, weight in self.action_keywords.items():
            if keyword in text:
                action_score += weight
        
        # Numerical parameters boost action score
        numbers = self.extract_numbers(text)
        if numbers:
            if any(freq_word in text for freq_word in ['hz', 'hertz', 'frequency']):
                action_score += 3
            if any(thick_word in text for thick_word in ['thick', 'thickness', 'm', 'ft']):
                action_score += 2
        
        # Decision logic with confidence threshold
        if question_score > action_score + 1:
            return 'question'
        elif action_score > question_score + 1:
            return 'action'
        else:
            return 'unclear'

    def extract_numbers(self, text: str) -> List[float]:
        """
        Extract numbers from text with unit handling.
        
        Args:
            text: Input text to extract numbers from
            
        Returns:
            List[float]: List of extracted numbers
        """
        number_patterns = [
            r'(\d+\.?\d*)\s*(?:hz|hertz)',
            r'(\d+\.?\d*)\s*(?:m|meter|metre|ft|feet|foot)',
            r'(\d+\.?\d*)'
        ]
        
        numbers = []
        for pattern in number_patterns:
            matches = re.findall(pattern, text.lower())
            numbers.extend([float(m) for m in matches])
        
        return list(set(numbers))  # Remove duplicates

    def extract_frequencies(self, text: str) -> List[float]:
        """
        Extract frequency values from text.
        
        Args:
            text: Input text to extract frequencies from
            
        Returns:
            List[float]: List of extracted frequencies
        """
        freq_patterns = [
            r'(\d+\.?\d*)\s*hz',
            r'frequency\s*(?:of|is|=|:)?\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*hertz',
            r'freq\s*(?:of|is|=|:)?\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*hz\s+(?:ricker|wavelet)',
            r'(?:ricker|wavelet)\s*(?:at|with|of)?\s*(\d+\.?\d*)\s*hz'
        ]
        
        frequencies = []
        for pattern in freq_patterns:
            matches = re.findall(pattern, text.lower())
            frequencies.extend([float(m) for m in matches])
        
        # If no specific frequency keywords, look for numbers in frequency context
        if not frequencies:
            numbers = self.extract_numbers(text)
            if numbers and any(keyword in text.lower() for keyword in ['frequency', 'freq', 'hz', 'hertz']):
                frequencies = [n for n in numbers if 1 <= n <= 1000]  # Reasonable frequency range
        
        return frequencies

    def normalize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameter keys to match tool requirements (e.g., map 'thickness' to 'max_thickness').
        """
        mapping = {
            'thickness': 'max_thickness',
            'thick': 'max_thickness',
            'max thickness': 'max_thickness',
            'maximum thickness': 'max_thickness',
        }
        normalized = {}
        for k, v in params.items():
            key = k.lower().replace('_', ' ')
            if key in mapping:
                normalized[mapping[key]] = v
            else:
                normalized[k] = v
        return normalized

    def extract_velocities(self, text: str) -> Dict[str, float]:
        """
        Extract velocities from text and map to vp1, vp2, vp3.
        Supports patterns like 'velocity[2000, 2500, 3000]', 'velocities: 2000, 2500, 3000', or 'vp1=2000, vp2=2500, vp3=3000'.
        """
        text = text.lower()
        velocities = {}
        # Pattern: velocity[2000, 2500, 3000] or velocities[...]
        match = re.search(r'velocit(?:y|ies)\s*\[([\d\s,]+)\]', text)
        if match:
            vals = [float(v.strip()) for v in match.group(1).split(',') if v.strip()]
            for i, v in enumerate(vals[:3]):
                velocities[f'vp{i+1}'] = v
        # Pattern: velocities: 2000, 2500, 3000
        match = re.search(r'velocit(?:y|ies)\s*[:=]\s*([\d\s,]+)', text)
        if match and not velocities:
            vals = [float(v.strip()) for v in match.group(1).split(',') if v.strip()]
            for i, v in enumerate(vals[:3]):
                velocities[f'vp{i+1}'] = v
        # Pattern: vp1=2000, vp2=2500, vp3=3000
        for i in range(1, 4):
            match = re.search(rf'vp{i}\s*[=:]\s*(\d+\.?\d*)', text)
            if match:
                velocities[f'vp{i}'] = float(match.group(1))
        return velocities

    def extract_densities(self, text: str) -> Dict[str, float]:
        """
        Extract densities from text and map to rho1, rho2, rho3.
        Supports patterns like 'density[2.1, 2.2, 2.3]', 'densities: 2.1, 2.2, 2.3', or 'rho1=2.1, rho2=2.2, rho3=2.3'.
        """
        text = text.lower()
        densities = {}
        # Pattern: density[2.1, 2.2, 2.3] or densities[...]
        match = re.search(r'densit(?:y|ies)\s*\[([\d\.\s,]+)\]', text)
        if match:
            vals = [float(v.strip()) for v in match.group(1).split(',') if v.strip()]
            for i, v in enumerate(vals[:3]):
                densities[f'rho{i+1}'] = v
        # Pattern: densities: 2.1, 2.2, 2.3
        match = re.search(r'densit(?:y|ies)\s*[:=]\s*([\d\.\s,]+)', text)
        if match and not densities:
            vals = [float(v.strip()) for v in match.group(1).split(',') if v.strip()]
            for i, v in enumerate(vals[:3]):
                densities[f'rho{i+1}'] = v
        # Pattern: rho1=2.1, rho2=2.2, rho3=2.3
        for i in range(1, 4):
            match = re.search(rf'rho{i}\s*[=:]\s*(\d+\.?\d*)', text)
            if match:
                densities[f'rho{i}'] = float(match.group(1))
        return densities
