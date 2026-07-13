#!/usr/bin/env python3
"""
Test script for the new tool use pattern implementation.
This demonstrates how the chatbot works with the tool use pattern from the notebook.
"""

import logging
from core.chatbot_tool_use import SeismicChatBotToolUse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_tool_use_pattern():
    """Test the tool use pattern implementation."""
    print("🌊 Testing Seismic ChatBot with Tool Use Pattern")
    print("=" * 50)
    
    # Initialize the chatbot
    chatbot = SeismicChatBotToolUse()
    
    # Test cases
    test_cases = [
        "What is a Ricker wavelet?",
        "Create a 30 Hz Ricker wavelet",
        "Make a wedge model with max_thickness=100, v1=2000, v2=2500, v3=3000, rho1=2.0, rho2=2.2, rho3=2.4",
        "Calculate Zoeppritz reflectivity for vp1=2000, vs1=800, rho1=2.0, vp2=2500, vs2=1000, rho2=2.2, angles=[0,10,20,30]",
        "quit"
    ]
    
    print("\nStarting interactive chat session...")
    print("Type 'quit' to exit\n")
    
    # Start the chat session
    chatbot.chat()

if __name__ == "__main__":
    test_tool_use_pattern() 