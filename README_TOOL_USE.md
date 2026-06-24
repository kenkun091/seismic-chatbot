# Seismic ChatBot - Tool Use Pattern Implementation

This document describes the refactored Seismic ChatBot that follows the tool use pattern from the notebook example.

## Overview

The chatbot has been refactored to use the same tool use pattern as demonstrated in the notebook, which provides:

1. **System Prompt with Tool Definitions**: Clear system prompt that defines available tools
2. **Tool Schema Definition**: Proper JSON schemas for each tool
3. **Tool Use Flow**: Proper message handling and tool execution flow
4. **Response Parsing**: XML tag-based response extraction

## Key Components

### 1. Tool Schemas (`config/tool_schemas.py`)

Defines all available tools with proper JSON schemas:

```python
TOOL_SCHEMAS = [
    {
        "name": "make_ricker",
        "description": "Creates a Ricker wavelet with specified frequency and time parameters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "frequency": {
                    "type": "number",
                    "description": "The dominant frequency of the Ricker wavelet in Hz."
                },
                # ... more properties
            },
            "required": ["frequency"]
        }
    },
    # ... more tools
]
```

### 2. Enhanced LLM Client (`core/llm_client.py`)

Updated to support tool use with proper message handling:

```python
def get_completion(self, system_prompt: str, user_prompt: str, 
                  tools: Optional[List[Dict]] = None, 
                  messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
    # Handles tool calls and returns structured response
```

### 3. Tool Manager (`core/tool_manager.py`)

Enhanced to process tool calls from the LLM:

```python
def process_tool_call(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
    # Validates and executes tools based on LLM requests
```

### 4. New ChatBot Implementation (`core/chatbot_tool_use.py`)

Main chatbot class that follows the notebook pattern:

```python
class SeismicChatBotToolUse:
    def chat(self, user_input: str = None) -> str:
        # Main chat loop following the notebook pattern
        
    def process_single_input(self, user_input: str) -> str:
        # Process single input for API interfaces
```

## Tool Use Flow

The implementation follows this flow:

1. **User Input**: User provides input
2. **LLM Analysis**: LLM analyzes input and decides whether to use tools
3. **Tool Execution**: If tools are needed, they are executed
4. **Result Processing**: Tool results are added to conversation
5. **Final Response**: LLM provides final response with tool results

## Usage Examples

### Command Line Interface

```bash
python test_tool_use.py
```

### Gradio Interface

```bash
python main.py
```

### Programmatic Usage

```python
from core.chatbot_tool_use import SeismicChatBotToolUse

chatbot = SeismicChatBotToolUse()

# Single input processing
response = chatbot.process_single_input("Create a 30 Hz Ricker wavelet")
print(response)

# Interactive chat
chatbot.chat()
```

## Example Interactions

### Creating a Ricker Wavelet
```
User: Create a 30 Hz Ricker wavelet
=====Using the make_ricker tool=====
Seismic Assistant: I've created a 30 Hz Ricker wavelet for you. The wavelet has been generated with the specified frequency and default time parameters. You can now use this wavelet for seismic modeling or analysis.
```

### Creating a Wedge Model
```
User: Make a wedge model with max_thickness=100, v1=2000, v2=2500, v3=3000, rho1=2.0, rho2=2.2, rho3=2.4
=====Using the wedge_model tool=====
Seismic Assistant: I've created a wedge model with the specified parameters. The model shows how seismic response varies with layer thickness, which is useful for understanding tuning effects and resolution limits.
```

### AVO Calculations
```
User: Calculate Zoeppritz reflectivity for vp1=2000, vs1=800, rho1=2.0, vp2=2500, vs2=1000, rho2=2.2, angles=[0,10,20,30]
=====Using the zoeppritz_reflectivity tool=====
Seismic Assistant: I've calculated the Zoeppritz reflectivity for your specified parameters. The results show how reflection coefficients vary with incidence angle, which is crucial for AVO analysis.
```

## Benefits of the Tool Use Pattern

1. **Clear Tool Definitions**: Each tool has a well-defined schema
2. **Automatic Parameter Extraction**: LLM can extract parameters from natural language
3. **Robust Error Handling**: Proper validation and error messages
4. **Context Awareness**: Tools can use context from previous interactions
5. **Extensible**: Easy to add new tools following the same pattern

## Migration from Old Implementation

The old implementation in `core/chatbot.py` is still available for backward compatibility. The new tool use pattern provides:

- Better parameter extraction from natural language
- More robust tool execution
- Clearer separation of concerns
- Better error handling and user feedback

## Configuration

The tool use pattern uses the same configuration files as the original implementation:

- `config/settings.py`: General settings and API configuration
- `config/tool_schemas.py`: Tool definitions and schemas
- Environment variables for API keys and endpoints

## Testing

Run the test script to verify the implementation:

```bash
python test_tool_use.py
```

This will start an interactive session where you can test various tool interactions. 