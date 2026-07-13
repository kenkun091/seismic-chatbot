import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.rag_tools import knowledge_rag

def test_general_rag():
    print("\n===== Testing General RAG Tool =====\n")
    
    # Test general query across all domains
    query = "What is a Ricker wavelet?"
    print(f"Query: {query}")
    results = knowledge_rag(query)
    
    print(f"\nFound {results['total_results']} results:")
    for i, result in enumerate(results['results']):
        print(f"\nResult {i+1}:")
        print(f"Domain: {result['domain']}")
        print(f"Topic: {result['topic']}")
        print(f"Score: {result['relevance_score']:.4f}")
        print(f"Content: {result['content'][:150]}...")
    
    # Test domain-specific query
    print("\n\n===== Testing Domain-Specific RAG =====\n")
    query = "How does porosity affect velocity?"
    domain = "rock_physics"
    print(f"Query: {query}")
    print(f"Domain: {domain}")
    
    results = knowledge_rag(query, domain=domain)
    
    print(f"\nFound {results['total_results']} results:")
    for i, result in enumerate(results['results']):
        print(f"\nResult {i+1}:")
        print(f"Domain: {result['domain']}")
        print(f"Topic: {result['topic']}")
        print(f"Score: {result['relevance_score']:.4f}")
        print(f"Content: {result['content'][:150]}...")

if __name__ == "__main__":
    test_general_rag()