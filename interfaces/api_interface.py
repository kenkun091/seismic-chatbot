import os
import time
import logging

from fastapi import FastAPI, HTTPException, Header, Request, Depends
from pydantic import BaseModel
from typing import List, Optional
from core.chatbot_tool_use import SeismicChatBotToolUse
from config.example_prompts import EXAMPLE_PROMPTS, search_prompts, get_random_prompts, get_prompts_by_category
from config.settings import LOG_LEVEL, LOG_FORMAT
from interfaces.security import RateLimiter, check_api_key

# No-op when main.py already configured the root logger; makes direct imports
# / uvicorn-style launches emit logs instead of silence.
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format=LOG_FORMAT)

from core.otel_export import install as _install_otel

_install_otel()  # no-op unless OTLP endpoint env vars are set

app = FastAPI(title="Seismic ChatBot API", description="API for seismic modeling assistant")

# Build the heavy components once; each request gets an isolated session so
# concurrent callers never share conversation context or token counters.
base_chatbot = SeismicChatBotToolUse()

# --- Security containment for the paid /chat endpoint -----------------------
# /chat proxies straight to a billed LLM. Require an API key (fail closed if the
# operator hasn't set one) and throttle per-client to bound cost/abuse.
API_AUTH_KEY = os.environ.get("API_AUTH_KEY")
_chat_rate_limiter = RateLimiter(
    max_requests=int(os.environ.get("CHAT_RATE_MAX", "30")),
    window_seconds=float(os.environ.get("CHAT_RATE_WINDOW_SECONDS", "60")),
)


def enforce_chat_policy(request: Request, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """Auth + rate-limit gate for /chat. Fails closed when no API key is configured."""
    if not API_AUTH_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server misconfigured: set API_AUTH_KEY to enable the /chat endpoint.",
        )
    if not check_api_key(x_api_key, API_AUTH_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")
    client = request.client.host if request.client else "unknown"
    if not _chat_rate_limiter.allow(client, now=time.time()):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")


# --- Session API for the outcrop web client ---------------------------------
from config.settings import SEISMIC_UPLOAD_DIR, MAX_IMAGE_MB
from interfaces.outcrop_api import build_router, install_error_handlers
from interfaces.sessions import SessionStore

# The /sessions routes (image/model/section/state/etc.) don't proxy to the billed
# LLM the way /chat does, so they get their own, separate rate budget rather than
# sharing (and being starved by, or starving) the /chat limiter.
_session_rate_limiter = RateLimiter(
    max_requests=int(os.environ.get("SESSION_RATE_MAX", "120")),
    window_seconds=float(os.environ.get("SESSION_RATE_WINDOW_SECONDS", "60")),
)


def enforce_session_policy(request: Request, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """Auth + rate-limit gate for /sessions routes. Same fail-closed logic as
    enforce_chat_policy, but throttled against its own limiter/budget."""
    if not API_AUTH_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server misconfigured: set API_AUTH_KEY to enable the /sessions endpoints.",
        )
    if not check_api_key(x_api_key, API_AUTH_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")
    client = request.client.host if request.client else "unknown"
    if not _session_rate_limiter.allow(client, now=time.time()):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")


session_store = SessionStore(
    base_chatbot,
    ttl_seconds=float(os.environ.get("SESSION_TTL_SECONDS", "7200")),
    max_sessions=int(os.environ.get("MAX_SESSIONS", "50")),
    upload_dir=SEISMIC_UPLOAD_DIR,
)
install_error_handlers(app)
app.include_router(build_router(session_store, enforce_session_policy,
                                SEISMIC_UPLOAD_DIR, MAX_IMAGE_MB))

# Static client bundle (webapp/dist) — mounted only when it has been built.
WEBAPP_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "webapp", "dist")
if os.path.isdir(WEBAPP_DIST):
    from fastapi.staticfiles import StaticFiles
    app.mount("/app", StaticFiles(directory=WEBAPP_DIST, html=True), name="webapp")


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    images: List[str] = []
    success: bool
    error: Optional[str] = None
    trace: Optional[dict] = None

class ExamplePrompt(BaseModel):
    title: str
    prompt: str
    description: str
    category: str

class PromptSearchRequest(BaseModel):
    query: str

class PromptSearchResponse(BaseModel):
    results: List[ExamplePrompt]
    total: int

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(enforce_chat_policy)])
async def chat(request: ChatRequest):
    """Process a chat message; return the narrated reply plus any plot paths."""
    try:
        session = base_chatbot.new_session()
        result = session.process_single_input(request.message)
        if isinstance(result, dict) and "reply" in result:
            return ChatResponse(
                response=str(result["reply"]),
                images=[str(p) for p in result.get("images") or []],
                success=True,
                trace=result.get("trace"),
            )
        return ChatResponse(response=str(result), success=True)
    except Exception as e:
        return ChatResponse(response="", success=False, error=str(e))

@app.get("/examples", response_model=dict)
async def get_all_examples():
    """Get all example prompts organized by category."""
    return EXAMPLE_PROMPTS

@app.get("/examples/categories", response_model=List[str])
async def get_categories():
    """Get all available categories."""
    return list(EXAMPLE_PROMPTS.keys())

@app.get("/examples/category/{category}", response_model=List[ExamplePrompt])
async def get_examples_by_category(category: str):
    """Get all examples for a specific category."""
    prompts = get_prompts_by_category(category)
    if not prompts:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
    
    return [
        ExamplePrompt(
            title=prompt["title"],
            prompt=prompt["prompt"],
            description=prompt["description"],
            category=category
        )
        for prompt in prompts
    ]

@app.post("/examples/search", response_model=PromptSearchResponse)
async def search_examples(request: PromptSearchRequest):
    """Search for example prompts."""
    results = search_prompts(request.query)
    
    return PromptSearchResponse(
        results=[
            ExamplePrompt(
                title=result["title"],
                prompt=result["prompt"],
                description=result["description"],
                category=result["category"]
            )
            for result in results
        ],
        total=len(results)
    )

@app.get("/examples/random", response_model=List[ExamplePrompt])
async def get_random_examples(count: int = 5):
    """Get random example prompts."""
    if count < 1 or count > 20:
        raise HTTPException(status_code=400, detail="Count must be between 1 and 20")
    
    results = get_random_prompts(count)
    
    return [
        ExamplePrompt(
            title=result["title"],
            prompt=result["prompt"],
            description=result["description"],
            category=result["category"]
        )
        for result in results
    ]

@app.get("/examples/copy/{category}/{index}")
async def copy_example_prompt(category: str, index: int):
    """Get a specific example prompt for copying."""
    prompts = get_prompts_by_category(category)
    if not prompts:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
    
    if index < 0 or index >= len(prompts):
        raise HTTPException(status_code=404, detail=f"Index {index} out of range")
    
    prompt = prompts[index]
    return {
        "title": prompt["title"],
        "prompt": prompt["prompt"],
        "description": prompt["description"],
        "category": category,
        "index": index
    }

if __name__ == "__main__":
    import uvicorn
    # Bind to localhost by default; override with API_HOST=0.0.0.0 only behind
    # a trusted proxy / once auth + rate limiting are confirmed in place.
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
