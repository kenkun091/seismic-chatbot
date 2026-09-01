import os
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL")

# Databricks Configuration (alternative to DeepSeek)
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
DATABRICKS_BASE_URL = os.environ.get("DATABRICKS_BASE_URL")

# Outcrop-photo upload sandbox (tools/image_safety.py). Absolute paths outside
# this directory are rejected by every image-consuming tool.
SEISMIC_UPLOAD_DIR = os.environ.get("SEISMIC_UPLOAD_DIR") or os.path.join(
    tempfile.gettempdir(), "seismic_uploads"
)
MAX_IMAGE_MB = float(os.environ.get("MAX_IMAGE_MB", "10"))

# Per-session decision-trace JSONL sink (core/turn_trace.py). One file per
# session at <SEISMIC_TRACE_DIR>/<session_id>.jsonl; SEISMIC_TRACE_DIR=off
# disables persistence.
_trace_dir_env = os.environ.get("SEISMIC_TRACE_DIR", "")
if _trace_dir_env.strip().lower() == "off":
    SEISMIC_TRACE_DIR = ""  # persistence disabled
else:
    SEISMIC_TRACE_DIR = _trace_dir_env or os.path.join(
        tempfile.gettempdir(), "seismic_traces"
    )

# RAG Configuration
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_TOP_K = 5
RAG_SIMILARITY_THRESHOLD = 0.3  # cosine similarity (collection uses cosine space)
RAG_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_VECTOR_DB_PATH = "./chroma_db"

# LLM Configuration
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 500

# Error Handling Configuration
MAX_ERRORS = 3

# Logging Configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Vision provider for outcrop-photo interpretation (core/vision_client.py).
# Optional: when nothing is set, interpret_outcrop raises a clear RuntimeError
# at call time and every other tool keeps working.
VISION_PROVIDER = os.environ.get("VISION_PROVIDER")          # "anthropic" | "openai" | None (auto)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
VISION_API_KEY = os.environ.get("VISION_API_KEY")            # OpenAI-compatible vision endpoint
VISION_BASE_URL = os.environ.get("VISION_BASE_URL")
VISION_MODEL = os.environ.get("VISION_MODEL")                # provider default when unset
