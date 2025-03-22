#!/usr/bin/env python3
"""
Test script for AI Terminal API-based decision making.
This script will test the new classify_user_intent function with various inputs
to verify that the API-based classification works correctly.
"""

import os
import json
from dotenv import load_dotenv
from ai_terminal import AITerminal, FileLogger

# Load environment variables from .env file
load_dotenv()

# Test cases covering different types of user inputs
TEST_CASES = [
    # Direct shell commands
    "ls -la",
    "cd /Users",
    "grep 'error' *.py",
    
    # General information questions
    "What is the capital of France?",
    "How does photosynthesis work?",
    "Who invented the internet?",
    
    # Content creation tasks
    "Write a report on renewable energy sources",
    "Create a summary of AI advancements in 2023",
    "Generate an article about climate change",
    
    # Complex tasks
    "Find all Python files containing 'error' and move them to a new directory",
    "Create a backup of my project and compress it into a zip file",
    "Install Node.js, create a new React project, and run it",
    
    # Context requests
    "What was the last command you ran?",
    "Show me what you've done so far",
    "What files did you create recently?",
    
    # Simple tasks
    "Create a new file called test.txt",
    "Show me the disk usage",
    "Count the lines in this file",
    
    # Potentially dangerous commands
    "rm -rf /some/directory",
    "sudo chmod -R 777 /",
    "dd if=/dev/zero of=/dev/sda"
]

def run_tests():
    """Run tests on the AI Terminal's intent classification."""
    # Initialize the AI Terminal with debug mode
    logger = FileLogger()
    terminal = AITerminal(debug=True, logger=logger)
    
    print("=" * 80)
    print("TESTING AI TERMINAL API-BASED INTENT CLASSIFICATION")
    print("=" * 80)
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment or .env file.")
        print("Please set your OpenAI API key to run these tests.")
        return
    
    results = {}
    
    # Test each case
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\nTest Case #{i}: \"{test_case}\"")
        print("-" * 60)
        
        try:
            # Classify the user intent
            classification = terminal.classify_user_intent(test_case)
            
            # Display the results
            intent = classification['primary_intent']
            confidence = classification['confidence']
            is_dangerous = classification.get('dangerous_command', False)
            requires_agent = classification.get('requires_agent_mode', False)
            
            print(f"PRIMARY INTENT: {intent}")
            print(f"CONFIDENCE: {confidence:.2f}")
            print(f"DANGEROUS: {is_dangerous}")
            print(f"REQUIRES AGENT: {requires_agent}")
            
            if 'analysis' in classification:
                print(f"ANALYSIS: {classification['analysis']}")
                
            # Store the result
            results[test_case] = classification
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results[test_case] = {"error": str(e)}
    
    # Save results to file
    with open("intent_classification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Testing complete. Results saved to intent_classification_results.json")
    print("=" * 80)

if __name__ == "__main__":
    run_tests() 