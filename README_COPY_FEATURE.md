# Copy Example Prompts Feature

This document describes the new copy functionality that allows users to easily copy example prompts directly from the interface.

## Overview

The copy feature provides users with a curated collection of example prompts that can be copied with a single click, making it easier to get started with the Seismic ChatBot.

## Features

### 1. Categorized Example Prompts

Example prompts are organized into logical categories:

- **Educational Questions**: Learn about seismic concepts and theory
- **Wavelet Tools**: Create and analyze Ricker wavelets
- **Wedge Modeling**: Build wedge models for seismic analysis
- **AVO Analysis**: Calculate reflectivity using various methods
- **Advanced Topics**: Complex seismic modeling scenarios

### 2. Copy Functionality

- **One-click copy**: Click the "📋 Copy" button to copy any prompt to clipboard
- **Visual feedback**: Button changes to "✅ Copied!" when successful
- **Notification**: Toast notification confirms successful copy
- **Fallback**: Manual copy option if clipboard API fails

### 3. Search and Filter

- **Search box**: Find specific examples by title, description, or prompt text
- **Real-time filtering**: Results update as you type
- **Category browsing**: Expand/collapse categories to browse examples

## Implementation

### Configuration File

The example prompts are defined in `config/example_prompts.py`:

```python
EXAMPLE_PROMPTS = {
    "Educational Questions": [
        {
            "title": "What is a Ricker wavelet?",
            "prompt": "What is a Ricker wavelet and why is it important in seismic analysis?",
            "description": "Learn about the fundamental wavelet used in seismic analysis"
        },
        # ... more examples
    ],
    # ... more categories
}
```

### Gradio Interface

The enhanced Gradio interface (`interfaces/gradio_interface.py`) includes:

- Accordion-style categorized examples
- Copy buttons that populate the main input field
- Search functionality for finding specific examples
- Responsive layout with better organization

### API Endpoints

The API interface (`interfaces/api_interface.py`) provides programmatic access:

- `GET /examples` - Get all examples by category
- `GET /examples/categories` - Get available categories
- `GET /examples/category/{category}` - Get examples for a category
- `POST /examples/search` - Search for examples
- `GET /examples/random` - Get random examples
- `GET /examples/copy/{category}/{index}` - Get specific example for copying

### Web Interface

A standalone HTML interface (`interfaces/web_interface.html`) demonstrates:

- Pure client-side copy functionality
- Modern, responsive design
- Search and filter capabilities
- Visual feedback for copy actions

## Usage Examples

### Gradio Interface

1. Launch the Gradio interface:
   ```bash
   python main.py
   ```

2. Browse categories in the right sidebar
3. Click "📋 Copy" next to any example
4. The prompt will be copied to the main input field
5. Click "Send" to execute the prompt

### Command Line Demo

Run the interactive demo:
```bash
python example_prompt_demo.py
```

Commands:
- `list` - Show all prompts by category
- `search` - Search for specific prompts
- `random` - Show random examples
- `quit` - Exit the demo

### API Usage

```python
import requests

# Get all examples
response = requests.get("http://localhost:8000/examples")
examples = response.json()

# Search for examples
response = requests.post("http://localhost:8000/examples/search", 
                        json={"query": "ricker"})
results = response.json()

# Get random examples
response = requests.get("http://localhost:8000/examples/random?count=3")
random_examples = response.json()
```

### Web Interface

1. Open `interfaces/web_interface.html` in a web browser
2. Use the search box to find specific examples
3. Click "📋 Copy Prompt" to copy any example
4. Paste the copied prompt into your chat interface

## Example Prompts Included

### Educational Questions
- What is a Ricker wavelet?
- How does frequency affect resolution?
- Explain tuning thickness
- What is AVO analysis?
- Zoeppritz vs Shuey equations

### Wavelet Tools
- Create 30 Hz Ricker wavelet
- Create 50 Hz Ricker with specific sampling
- Plot wavelet spectrum

### Wedge Modeling
- Simple wedge model with parameters
- Gas sand wedge modeling
- Oil sand wedge modeling

### AVO Analysis
- Zoeppritz reflectivity calculation
- Shuey reflectivity calculation
- Plot AVO curves
- Gas sand AVO modeling

### Advanced Topics
- Tuning analysis
- Resolution limits
- AVO classification

## Benefits

1. **User Experience**: Easy access to working examples
2. **Learning**: Users can see what's possible and how to phrase requests
3. **Efficiency**: No need to remember or type complex parameter strings
4. **Discovery**: Users can explore different types of analysis
5. **Consistency**: Standardized examples ensure reproducible results

## Technical Details

### Clipboard API

The copy functionality uses the browser's Clipboard API:

```javascript
navigator.clipboard.writeText(text).then(() => {
    showNotification();
    updateButtonText();
}).catch(err => {
    fallbackToManualCopy();
});
```

### Fallback Mechanism

If the Clipboard API is not available, the interface provides:

- Manual copy instructions
- Highlighted text for easy selection
- Clear visual indicators

### Responsive Design

The interface adapts to different screen sizes:

- Desktop: Sidebar layout with categories
- Mobile: Stacked layout with collapsible sections
- Tablet: Hybrid layout with touch-friendly buttons

## Future Enhancements

1. **User Custom Examples**: Allow users to save their own examples
2. **Example Ratings**: Let users rate helpful examples
3. **Context-Aware Suggestions**: Suggest examples based on conversation history
4. **Export/Import**: Share example collections between users
5. **Integration**: Direct integration with other seismic software

## Dependencies

- `pyperclip` (optional): For command-line clipboard functionality
- `gradio`: For the web interface
- `fastapi`: For the API interface
- Modern browser: For web interface clipboard API

## Installation

The copy feature is included by default. For enhanced clipboard support:

```bash
pip install pyperclip
```

## Testing

Test the copy functionality:

```bash
# Test Gradio interface
python main.py

# Test command-line demo
python example_prompt_demo.py

# Test API (if FastAPI is installed)
python interfaces/api_interface.py
```

The copy feature enhances the user experience by providing easy access to working examples, making the Seismic ChatBot more accessible and user-friendly. 