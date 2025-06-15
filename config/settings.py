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
        'optional_params': {'dt': 0.001, 'duration': 0.256}
    },
    'plot_ricker': {
        'description': 'Plots a Ricker wavelet with time domain and frequency domain analysis',
        'keywords': ['plot', 'show', 'visualize', 'display', 'graph', 'chart'],
        'required_params': ['wavelet'],
        'optional_params': {'time': None}
    },
    'wedge_model': {
        'description': 'Creates a wedge model for seismic analysis with variable thickness',
        'keywords': ['wedge', 'model', 'seismic', 'thickness', 'layer', 'synthetic', 'modeling'],
        'required_params': ['max_thickness', 'vp1', 'vp2', 'vp3', 'rho1', 'rho2', 'rho3'],
        'optional_params': {
            'zunit': 'm',
            'wv_type': 'ricker',
            'ricker_freq': 30,
            'ormsby_freq': "",
            'wavelet_str': "",
            'wavelet_fname': "",
            'phase_rot': 0,
            'gain': 1.0,
            'plotpadtime': 100,
            'thickness_domain': 'depth',
            'fig_fname': 'wedge_model.png',
            'csv_fname': 'wedge_data.csv'
        }
    }
}

# Error Handling Configuration
MAX_ERRORS = 3

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
