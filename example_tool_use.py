#!/usr/bin/env python3
"""
Example script demonstrating the tool use pattern implementation.
This shows how the chatbot handles different types of requests.
"""

import logging
from core.chatbot_tool_use import SeismicChatBotToolUse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def demonstrate_tool_use():
    """Demonstrate the tool use pattern with various examples."""
    print("🌊 Seismic ChatBot Tool Use Pattern Demonstration")
    print("=" * 60)
    
    # Initialize the chatbot
    chatbot = SeismicChatBotToolUse()
    
    # Example 1: Educational question (no tool use)
    print("\n📚 Example 1: Educational Question")
    print("-" * 40)
    question = "What is a Ricker wavelet and why is it important in seismic analysis?"
    print(f"User: {question}")
    response = chatbot.process_single_input(question)
    print(f"Assistant: {response}")
    
    # Example 2: Tool use - Create Ricker wavelet
    print("\n🔧 Example 2: Tool Use - Create Ricker Wavelet")
    print("-" * 40)
    request = "Create a 30 Hz Ricker wavelet"
    print(f"User: {request}")
    response = chatbot.process_single_input(request)
    print(f"Assistant: {response}")
    
    # Example 3: Tool use - Wedge model with parameters
    print("\n🔧 Example 3: Tool Use - Wedge Model")
    print("-" * 40)
    request = "Make a wedge model with max_thickness=100, v1=2500, v2=2400, v3=2500, rho1=2.2, rho2=2.1, rho3=2.2"
    print(f"User: {request}")
    response = chatbot.process_single_input(request)
    print(f"Assistant: {response}")
    
    # Example 4: Tool use - AVO calculation
    print("\n🔧 Example 4: Tool Use - AVO Calculation")
    print("-" * 40)
    request = "Calculate Zoeppritz reflectivity for vp1=2000, vs1=800, rho1=2.0, vp2=2500, vs2=1000, rho2=2.2, angles=[0,10,20,30]"
    print(f"User: {request}")
    response = chatbot.process_single_input(request)
    print(f"Assistant: {response}")
    
    # Example 5: Follow-up question using context
    print("\n🔄 Example 5: Follow-up Question")
    print("-" * 40)
    request = "Now plot that wavelet"
    print(f"User: {request}")
    response = chatbot.process_single_input(request)
    print(f"Assistant: {response}")
    
    print("\n✅ Demonstration completed!")
    print("\nKey Benefits of Tool Use Pattern:")
    print("1. Natural language parameter extraction")
    print("2. Automatic tool selection based on user intent")
    print("3. Context awareness across interactions")
    print("4. Robust error handling and validation")
    print("5. Clear separation between educational and computational tasks")

if __name__ == "__main__":
    demonstrate_tool_use() 