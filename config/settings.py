import os
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

# Error Handling Configuration
MAX_ERRORS = 3

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
