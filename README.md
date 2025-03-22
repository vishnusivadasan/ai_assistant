# AI Terminal Agent

AI Terminal Agent is a command-line tool that uses AI to convert natural language instructions into shell commands, execute them, and handle errors intelligently.

## Features

- Convert natural language to shell commands
- Execute commands with permission checks
- Intelligent error handling
- Conversation history and logging
- Support for complex, multi-step tasks
- Safety checks for potentially dangerous commands
- Command aliases for frequently used queries
- Flexible environment configuration
- Shell completion for bash, zsh, and fish

## Installation

You can install the AI Terminal Agent using pip:

```bash
pip install ai-terminal-agent
```

For development installation:

```bash
git clone https://github.com/yourusername/ai_terminal.git
cd ai_terminal
pip install -e .
```

### Shell Completion

For tab completion support in your shell, run one of the following commands:

For Bash:
```bash
agent-completion bash --install
```

For Zsh:
```bash
agent-completion zsh --install
```

For Fish:
```bash
agent-completion fish --install
```

You can also manually add completion to your shell by running:
```bash
# For bash
agent-completion bash >> ~/.bash_completion

# For zsh (add to your .zshrc)
agent-completion zsh > ~/.zsh/completion/_agent
# Make sure to add to fpath in .zshrc: fpath=(~/.zsh/completion $fpath)

# For fish
agent-completion fish > ~/.config/fish/completions/agent.fish
```

## Setup

1. Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
DEFAULT_MODE=manual
```

2. Make sure you have the required environment variables set.

## Usage

You can use the AI Terminal Agent in two ways:

### One-shot mode

Simply provide your query as a command-line argument:

```bash
agent "show me all running docker containers"
```

```bash
agent "find all python files in the current directory modified in the last week"
```

### Interactive mode

Run the agent without any arguments to enter interactive mode:

```bash
agent
```

### Command aliases

You can save frequently used commands as aliases:

```bash
# Save an alias
agent --save-alias docker_ps "show me all running docker containers"

# Use an alias
agent --use-alias docker_ps

# List all aliases
agent --list-aliases

# Delete an alias
agent --delete-alias docker_ps
```

### Environment configuration

You can specify a custom environment file:

```bash
# Use a specific environment file
agent --env-file .env.production "deploy the application"
```

### Command-line options

```
usage: agent [-h] [--save-alias NAME] [--list-aliases] [--use-alias NAME]
             [--delete-alias NAME] [--mode {manual,autonomous}] [--env-file ENV_FILE]
             [--debug] [--max-plan-iterations MAX_PLAN_ITERATIONS]
             [--no-direct-execution] [--no-refine-queries] [--show-log]
             [query]

AI-Powered Terminal Agent

positional arguments:
  query                 Query or task for the AI agent (in quotes)

options:
  -h, --help            show this help message and exit

Alias Management:
  --save-alias NAME     Save the provided query as an alias with the given name
  --list-aliases        List all available command aliases
  --use-alias NAME      Run the query associated with the given alias name
  --delete-alias NAME   Delete the specified alias

  --mode {manual,autonomous}
                        Execution mode - 'manual' or 'autonomous'
  --env-file ENV_FILE   Path to the environment file (default: .env)
  --debug               Enable debug logging
  --max-plan-iterations MAX_PLAN_ITERATIONS
                        Maximum number of iterations for plan verification
  --no-direct-execution
                        Disable direct execution of shell commands
  --no-refine-queries   Disable query refinement via API
  --show-log            Display the log after completion when running in one-
                        shot mode
```

## Examples

```bash
# Simple file operations
agent "create a new directory called 'project' and initialize a git repository there"

# Finding information
agent "what's my IP address and current location?"

# Advanced tasks
agent "download the latest nodejs LTS, install it, and create a sample express app"

# With options
agent --mode autonomous "organize my downloads folder by file type"

# Using aliases
agent --save-alias cleanup "find and delete all temporary files in the current directory"
agent --use-alias cleanup

# Using custom environment file
agent --env-file .env.dev "deploy to development environment"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
