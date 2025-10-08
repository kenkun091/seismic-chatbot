import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL")

# Databricks Configuration (alternative to DeepSeek)
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
DATABRICKS_BASE_URL = os.environ.get("DATABRICKS_BASE_URL")

# RAG Configuration
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_TOP_K = 5
RAG_SIMILARITY_THRESHOLD = 0.7
RAG_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_VECTOR_DB_PATH = "./chroma_db"

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
            'zunit': 'm',
            'vs1': None,
            'vs2': None,
            'vs3': None,
            'incident_angle': 0,
        }
    },
    'plot_wedge_model': {
        'description': 'Plots a wedge model showing seismic response vs thickness',
        'keywords': ['plot', 'show', 'visualize', 'display', 'graph', 'chart', 'wedge', 'enlarge', 'bigger', 'larger'],
        'required_params': ['synthetic_data', 'parameters'],
        'optional_params': {'figsize': [12, 14]}
    }
}

# Error Handling Configuration
MAX_ERRORS = 3

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
