from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from core.chatbot_tool_use import SeismicChatBotToolUse
from config.example_prompts import EXAMPLE_PROMPTS, search_prompts, get_random_prompts, get_prompts_by_category

app = FastAPI(title="Seismic ChatBot API", description="API for seismic modeling assistant")

# Initialize the chatbot
chatbot = SeismicChatBotToolUse()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None

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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return the response."""
    try:
        response = chatbot.process_single_input(request.message)
        return ChatResponse(response=str(response), success=True)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
