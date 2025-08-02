#!/usr/bin/env python3
"""
Demo script for the example prompts feature.
Shows how users can browse and copy example prompts.
"""

import pyperclip
from config.example_prompts import EXAMPLE_PROMPTS, search_prompts, get_random_prompts

def display_prompts():
    """Display all available prompts in a user-friendly format."""
    print("🌊 Seismic ChatBot - Example Prompts")
    print("=" * 50)
    
    for category, prompts in EXAMPLE_PROMPTS.items():
        print(f"\n📂 {category}")
        print("-" * len(category))
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}. {prompt['title']}")
            print(f"   Description: {prompt['description']}")
            print(f"   Prompt: {prompt['prompt']}")
            print(f"   [Press Enter to copy this prompt]")
            
            # Wait for user input
            input()
            
            # Copy to clipboard
            pyperclip.copy(prompt['prompt'])
            print(f"   ✅ Copied to clipboard!")
            print()

def search_demo():
    """Demonstrate search functionality."""
    print("\n🔍 Search Demo")
    print("=" * 20)
    
    while True:
        query = input("Enter search term (or 'quit' to exit): ").strip()
        
        if query.lower() == 'quit':
            break
            
        if not query:
            print("Showing random examples:")
            results = get_random_prompts(3)
        else:
            print(f"Searching for: '{query}'")
            results = search_prompts(query)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} ({result['category']})")
                print(f"   {result['description']}")
                print(f"   Prompt: {result['prompt']}")
        else:
            print("No results found.")

def interactive_demo():
    """Interactive demo with copy functionality."""
    print("🌊 Seismic ChatBot - Interactive Example Prompts")
    print("=" * 55)
    print("This demo shows how users can browse and copy example prompts.")
    print("Commands:")
    print("  'list' - Show all prompts by category")
    print("  'search' - Search for specific prompts")
    print("  'random' - Show random examples")
    print("  'quit' - Exit the demo")
    print()
    
    while True:
        command = input("Enter command: ").strip().lower()
        
        if command == 'quit':
            print("Goodbye!")
            break
        elif command == 'list':
            display_prompts()
        elif command == 'search':
            search_demo()
        elif command == 'random':
            print("\n🎲 Random Examples:")
            print("-" * 20)
            results = get_random_prompts(5)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} ({result['category']})")
                print(f"   {result['description']}")
                print(f"   Prompt: {result['prompt']}")
                print(f"   [Press Enter to copy]")
                input()
                pyperclip.copy(result['prompt'])
                print(f"   ✅ Copied to clipboard!")
        else:
            print("Unknown command. Try 'list', 'search', 'random', or 'quit'.")

if __name__ == "__main__":
    try:
        interactive_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
    except ImportError:
        print("Error: pyperclip not installed. Install it with: pip install pyperclip")
        print("Running without clipboard functionality...")
        
        # Fallback without clipboard
        print("\n🌊 Seismic ChatBot - Example Prompts (Read-only)")
        print("=" * 55)
        
        for category, prompts in EXAMPLE_PROMPTS.items():
            print(f"\n📂 {category}")
            print("-" * len(category))
            
            for i, prompt in enumerate(prompts, 1):
                print(f"\n{i}. {prompt['title']}")
                print(f"   Description: {prompt['description']}")
                print(f"   Prompt: {prompt['prompt']}") 