# AI Terminal

An intelligent, AI-powered command-line interface that uses natural language processing to convert your plain English requests into shell commands, execute them, and help you accomplish tasks more efficiently.

## Features

- **Natural Language Command Generation**: Ask for what you want in plain English
- **Intelligent Error Handling**: The AI analyzes errors and suggests fixes
- **Context Awareness**: The terminal remembers previous commands and their results
- **Content Creation**: Generate reports, articles, and documentation directly
- **General Knowledge**: Ask general information questions without leaving the terminal
- **Complex Task Planning**: Break down complex tasks into executable steps
- **Command Safety Verification**: Checks if commands might be potentially dangerous
- **Conversation Logging**: All interactions are logged for future reference

## Installation

### Prerequisites

- Python 3.8 or higher
- An OpenAI API key (with access to GPT-4 or newer models)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/vishnusivadasan/ai_assistant.git
   cd ai_assistant
   ```

2. Install the required packages:
   ```bash
   pip install openai python-dotenv rich psutil
   ```

3. Create a `.env` file in the project directory:
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   echo "OPENAI_MODEL=gpt-4-turbo-preview" >> .env
   ```
   Replace `your-api-key-here` with your actual OpenAI API key.

## Usage

### Basic Usage

Run the AI Terminal in interactive mode:

```bash
python ai_terminal.py
```

This will start an interactive session where you can type requests in natural language:

```
What would you like me to do? Find all Python files in this directory
```

### Example Requests

Here are some examples of what you can ask AI Terminal to do:

#### Command Generation
- "Find all Python files containing 'error' and move them to a new directory"
- "Check disk usage and show the 5 largest files"
- "Create a backup of this project and compress it into a zip file"

#### General Information
- "What is the capital of France?"
- "How does photosynthesis work?"
- "Who invented the internet?"

#### Content Creation
- "Write a report on renewable energy sources"
- "Create a summary of AI advancements in 2023"
- "Generate detailed documentation for this project"

#### Context Requests
- "What was the last command you ran?"
- "Show me what you've done so far"
- "What files did you create recently?"

### Command-Line Arguments

AI Terminal supports several command-line arguments:

```
usage: ai_terminal.py [-h] [--mode {manual,autonomous}] [--debug] [--max-plan-iterations MAX_PLAN_ITERATIONS]
                      [--no-direct-execution] [--no-refine-queries] [--max-logs MAX_LOGS]
                      [--logs-dir LOGS_DIR] [--request REQUEST] [--show-log]
```

| Argument | Description |
|----------|-------------|
| `--mode {manual,autonomous}` | Execution mode (default: manual) |
| `--debug` | Enable debug logging |
| `--max-plan-iterations N` | Maximum iterations for plan verification (default: 1) |
| `--no-direct-execution` | Disable direct execution of shell commands |
| `--no-refine-queries` | Disable query refinement via API |
| `--max-logs N` | Maximum number of log files to keep |
| `--logs-dir DIR` | Directory where conversation logs are stored |
| `--request "text"` | Run in one-shot mode with the provided request |
| `--show-log` | Display the log after completion in one-shot mode |

### Examples Using Arguments

One-shot mode (execute a single request and exit):
```bash
python ai_terminal.py --request "Find all files modified in the last 24 hours"
```

Debug mode with increased plan iterations:
```bash
python ai_terminal.py --debug --max-plan-iterations 3
```

Run in autonomous mode with custom logs directory:
```bash
python ai_terminal.py --mode autonomous --logs-dir ~/ai_terminal_logs
```

## Modes

### Manual Mode (Default)
- The AI suggests commands and asks for confirmation only if they appear dangerous
- You maintain control over execution, especially for potentially risky operations

### Autonomous Mode
- Similar to manual mode but with more independence in executing multi-step tasks
- Still asks for confirmation for potentially dangerous operations

## Testing

The project includes a comprehensive test suite in the `tests/` directory:

```bash
cd tests
python test_ai_terminal.py       # Test intent classification
python test_command_danger.py    # Test command danger assessment
python test_process_request.py   # Test full request processing
```

See the `tests/TESTING.md` file for detailed testing instructions.

## Troubleshooting

### API Key Issues
- Ensure your OpenAI API key is correctly set in the `.env` file
- Check that you have sufficient API credits and quota

### Command Execution Problems
- If commands fail, try running in debug mode with the `--debug` flag
- Check the log files in the logs directory (default: `~/.ai_terminal_logs/`)

### Viewing Logs
- Use the command `show log` in the terminal to view the current log
- Use `show logs` to list all available logs
- Use `open logs` to open the logs directory in your file explorer

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
