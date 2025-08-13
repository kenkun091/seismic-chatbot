#!/usr/bin/env python3
"""
Test script to verify token usage tracking functionality.
"""

import logging
from core.chatbot_tool_use import SeismicChatBotToolUse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_token_usage():
    """Test the token usage tracking functionality."""
    print("🌊 Testing Token Usage Tracking")
    print("=" * 50)
    
    # Initialize the chatbot
    chatbot = SeismicChatBotToolUse()
    
    # Process a simple query
    print("\nProcessing query: 'What is a Ricker wavelet?'")
    response = chatbot.process_single_input("What is a Ricker wavelet?")
    print(f"\nResponse: {response}")
    
    # Get and display token usage
    token_usage = chatbot.context_manager.get_token_usage()
    print("\nToken Usage:")
    print(f"  Prompt Tokens: {token_usage['prompt_tokens']}")
    print(f"  Completion Tokens: {token_usage['completion_tokens']}")
    print(f"  Total Tokens: {token_usage['total_tokens']}")
    
    # Process another query to see cumulative usage
    print("\nProcessing query: 'Create a 30 Hz Ricker wavelet'")
    response = chatbot.process_single_input("Create a 30 Hz Ricker wavelet")
    print(f"\nResponse: {response}")
    
    # Get and display updated token usage
    token_usage = chatbot.context_manager.get_token_usage()
    print("\nUpdated Token Usage:")
    print(f"  Prompt Tokens: {token_usage['prompt_tokens']}")
    print(f"  Completion Tokens: {token_usage['completion_tokens']}")
    print(f"  Total Tokens: {token_usage['total_tokens']}")
    
    # Reset context and verify token usage is cleared
    print("\nResetting context (simulating browser refresh)...")
    chatbot.context_manager.clear_context()
    
    # Verify token usage is reset
    token_usage = chatbot.context_manager.get_token_usage()
    print("\nToken Usage After Reset:")
    print(f"  Prompt Tokens: {token_usage['prompt_tokens']}")
    print(f"  Completion Tokens: {token_usage['completion_tokens']}")
    print(f"  Total Tokens: {token_usage['total_tokens']}")

if __name__ == "__main__":
    test_token_usage()