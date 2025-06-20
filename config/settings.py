import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL")

# LLM Configuration
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 500

# Tool Configuration
AVAILABLE_TOOLS: Dict[str, Dict[str, Any]] = {
    'make_ricker': {
        'description': 'Creates a Ricker wavelet',
        'keywords': ['ricker', 'wavelet', 'create', 'make', 'generate'],
        'required_params': ['frequency'],
        'optional_params': {'dt': 0.001, 'time_length': 256}
    },
    'plot_ricker': {
        'description': 'Plots a Ricker wavelet with time domain and frequency domain analysis',
        'keywords': ['plot', 'show', 'visualize', 'display', 'graph', 'chart'],
        'required_params': ['wavelet'],
        'optional_params': {'time_array': None}
    },
    'wedge_model': {
        'description': 'Creates a wedge model for seismic analysis with variable thickness',
        'keywords': ['wedge', 'model', 'seismic', 'thickness', 'layer', 'synthetic', 'modeling'],
        'required_params': ['max_thickness', 'v1', 'v2', 'v3', 'rho1', 'rho2', 'rho3'],
        'optional_params': {
            'num_traces': 61,
            'dt': 0.1,
            'wavelet_freq': 30.0,
            'wavelet_length': 256.0,
            'phase_rot': 0.0,
            'wv_type': 'ricker',
            'ormsby_freq': None,
            'gain': 1.0,
            'plotpadtime': 50.0,
            'thickness_domain': 'depth',
            'zunit': 'm'
        }
    }
}

# Error Handling Configuration
MAX_ERRORS = 3

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
