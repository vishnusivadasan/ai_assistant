# AI-Powered Terminal

A Python-based AI-powered terminal application that interacts with the OpenAI API to convert natural language instructions into shell commands, execute them, and handle errors intelligently.

## Features

- Convert natural language instructions into shell commands
- Execute commands in macOS Terminal (zsh/bash)
- Support for Agent Mode to figure out execution steps automatically
- Debug errors and retry commands when necessary
- Memory to maintain context during debugging
- Manual and autonomous execution modes
- Analyze command output logs to decide next actions

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ai_terminal.git
   cd ai_terminal
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Copy the example environment file and add your OpenAI API key:
   ```
   cp .env.example .env
   ```
   Edit `.env` and add your OpenAI API key.

## Usage

Run the AI Terminal with:

```
python ai_terminal.py
```

### Command-line Arguments

- `--mode`: Set the execution mode (manual or autonomous). Default is manual.
  ```
  python ai_terminal.py --mode autonomous
  ```

- `--debug`: Enable debug logging.
  ```
  python ai_terminal.py --debug
  ```

### Execution Modes

1. **Manual Mode (Default)**: 
   - The AI suggests commands and asks for confirmation before execution.

2. **Autonomous Mode**:
   - Runs commands automatically without asking for approval.
   - Use with caution as it can execute commands without confirmation.

### Example Queries

- "Find large files in my Downloads folder"
- "Set up a Python virtual environment with Flask"
- "Create a new directory structure for a React project"
- "Check what's using port 3000 and kill it"

## Security Considerations

- The AI Terminal will ask for confirmation before executing potentially destructive commands.
- Commands requiring elevated privileges will explicitly require approval.
- Always review commands in manual mode before allowing execution.

## License

MIT 