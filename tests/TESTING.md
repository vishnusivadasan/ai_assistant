# Testing AI Terminal API-Based Decision Making

This document provides instructions for testing the new API-based decision making features in the AI Terminal application.

## Prerequisites

Before running the tests, ensure you have:

1. An OpenAI API key with access to the required models
2. The API key set in your environment or in a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   OPENAI_MODEL=gpt-4-turbo-preview  # or another supported model
   ```
3. All required dependencies installed:
   ```
   pip install openai python-dotenv rich psutil
   ```

## Available Test Scripts

We've provided three test scripts to verify different aspects of the API-based decision making. All test scripts are located in the `tests/` directory.

### 1. Intent Classification Test (`tests/test_ai_terminal.py`)

This script tests how the AI classifies different types of user input, including:
- Direct shell commands
- General information questions
- Content creation tasks
- Complex tasks
- Context requests
- Simple tasks
- Potentially dangerous commands

Run the test:
```
cd tests
python test_ai_terminal.py
```

The results will be displayed in the terminal and saved to `intent_classification_results.json` for review.

### 2. Command Danger Assessment Test (`tests/test_command_danger.py`)

This script tests the AI's ability to assess whether commands are dangerous, including:
- Safe commands
- Potentially risky commands
- Known dangerous commands

Run the test:
```
cd tests
python test_command_danger.py
```

The results will be displayed in the terminal and saved to `command_danger_results.json` for review.

### 3. Full Request Processing Test (`tests/test_process_request.py`)

This script tests the complete request processing flow to verify that the API-based classification leads to correct handling of different types of requests.

Run the test:
```
cd tests
python test_process_request.py
```

This test is interactive and will pause after each request to allow you to review the results.

## What to Look For

When evaluating the test results, pay attention to:

1. **Classification Accuracy**: Are user inputs correctly classified into the appropriate intent categories?

2. **Confidence Scores**: How confident is the model in its classifications? Higher scores (closer to 1.0) indicate more confidence.

3. **Danger Assessment**: Are dangerous commands correctly identified? Are there any false positives (safe commands identified as dangerous) or false negatives (dangerous commands identified as safe)?

4. **Pathway Selection**: Does the system select the appropriate processing pathway based on the classification?

5. **API Usage Efficiency**: How many API calls are made for each request? Could any be consolidated or eliminated?

## Expected Results

The expected results for each test type are:

### Intent Classification:
- Direct shell commands should be classified as "direct_command"
- General information questions should be classified as "general_information"
- Content creation tasks should be classified as "content_creation"
- Complex tasks should be classified as "complex_task" with "requires_agent_mode" often set to true
- Context requests should be classified as "context_request"
- Simple tasks should be classified as "simple_task"

### Command Danger Assessment:
- Safe commands should have "is_dangerous" = false and low risk levels (0-3)
- Potentially risky commands should have moderate risk levels (4-7)
- Known dangerous commands should have "is_dangerous" = true and high risk levels (8-10)

### Full Request Processing:
- General information questions should be answered directly without command execution
- Content creation tasks should be handled by agent_mode with content creation focus
- Context requests should search conversation history and provide relevant information
- Direct commands should be executed directly (with confirmation if dangerous)
- Complex tasks should be broken down and executed step by step
- Simple tasks should be converted to a single command and executed

## Troubleshooting

If you encounter issues during testing:

1. **API Key Issues**: Ensure your OpenAI API key is valid and has sufficient quota

2. **Model Availability**: Make sure you're using a model that's available to you

3. **Classification Errors**: If classification results are inconsistent, try adjusting the classification prompt in the `classify_user_intent` method

4. **Dangerous Command Detection Issues**: You may need to adjust the prompt in the `check_command_danger_with_ai` method

5. **Debug Mode**: Enable debug mode for more verbose logging 