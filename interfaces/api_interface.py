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
