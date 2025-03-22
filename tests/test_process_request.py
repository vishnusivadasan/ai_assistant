#!/usr/bin/env python3
"""
Test script for AI Terminal process_request function with API-based decision making.
This script tests the full request processing flow to verify that the API-based
classification leads to the correct handling of different types of requests.
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to sys.path to import the ai_terminal module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_terminal import AITerminal, FileLogger

# Load environment variables from .env file
load_dotenv()

# Test cases for different request types
TEST_REQUESTS = [
    # General information questions
    "What is the capital of France?",
    
    # Content creation tasks
    "Write a brief summary of the AI Terminal project",
    
    # Context requests
    "What was the last command we ran?",
    
    # Direct commands (if direct_execution is enabled)
    "ls -la",
    
    # Complex tasks
    "Find all text files in the current directory and count the lines in each",
    
    # Simple tasks
    "Create a file called testfile.txt with the text 'Hello, World!'"
]

def run_tests():
    """Run tests on the AI Terminal process_request function."""
    # Initialize the AI Terminal with debug mode and direct execution
    logger = FileLogger()
    terminal = AITerminal(
        mode="manual",
        debug=True, 
        logger=logger,
        direct_execution=True,
        refine_queries=True,
        max_plan_iterations=1
    )
    
    print("=" * 80)
    print("TESTING AI TERMINAL REQUEST PROCESSING WITH API-BASED DECISION MAKING")
    print("=" * 80)
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment or .env file.")
        print("Please set your OpenAI API key to run these tests.")
        return
    
    # Process each request and observe handling
    for i, request in enumerate(TEST_REQUESTS, 1):
        print("\n" + "=" * 80)
        print(f"Test Request #{i}: \"{request}\"")
        print("-" * 60)
        
        try:
            # Process the request in one-shot mode for more verbose output
            terminal.process_request(request, one_shot_mode=True)
            
            print("\nRequest processing completed.")
            print("-" * 60)
            print(f"Check the log file for details: {terminal.logger.log_file}")
            
        except Exception as e:
            print(f"ERROR processing request: {str(e)}")
            
        # Wait for user to continue
        input("\nPress Enter to continue to the next test case...")
    
    print("\n" + "=" * 80)
    print(f"Testing complete. Log file: {terminal.logger.log_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_tests() 