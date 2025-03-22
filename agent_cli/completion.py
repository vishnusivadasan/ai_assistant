#!/usr/bin/env python3
"""
Shell completion for the agent command.
"""

import os
import sys
from pathlib import Path

# Add parent directory to sys.path to import cli
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cli import load_aliases

BASH_COMPLETION = """
_agent_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # All options
    opts="--mode --debug --max-plan-iterations --no-direct-execution --no-refine-queries --show-log --save-alias --list-aliases --use-alias --delete-alias --env-file"
    
    # Handle special cases
    case "${prev}" in
        --mode)
            COMPREPLY=( $(compgen -W "manual autonomous" -- ${cur}) )
            return 0
            ;;
        --env-file)
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
        --use-alias|--delete-alias)
            # Use the available aliases for completion
            local aliases=$(agent --list-aliases | grep -v "No aliases" | grep -v "Available" | sed 's/^[[:space:]]*//g' | cut -d ':' -f 1)
            COMPREPLY=( $(compgen -W "${aliases}" -- ${cur}) )
            return 0
            ;;
        *)
            ;;
    esac
    
    # Complete options if cur starts with -
    if [[ ${cur} == -* ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
    
    return 0
}

complete -F _agent_completion agent
"""

ZSH_COMPLETION = """
#compdef agent

_agent() {
    local -a commands
    local -a options
    
    options=(
        '--mode[Execution mode]:mode:(manual autonomous)'
        '--debug[Enable debug logging]'
        '--max-plan-iterations[Maximum number of iterations for plan verification]:iterations:'
        '--no-direct-execution[Disable direct execution of shell commands]'
        '--no-refine-queries[Disable query refinement via API]'
        '--show-log[Display the log after completion]'
        '--save-alias[Save the provided query as an alias]:alias name:'
        '--list-aliases[List all available command aliases]'
        '--use-alias[Run the query associated with the given alias name]:alias name:->aliases'
        '--delete-alias[Delete the specified alias]:alias name:->aliases'
        '--env-file[Path to the environment file]:env file:_files'
    )
    
    _arguments -C $options '*::args:->args'
    
    case $state in
        aliases)
            local -a aliases
            aliases=($(agent --list-aliases | grep -v "No aliases" | grep -v "Available" | sed 's/^[[:space:]]*//g' | cut -d ':' -f 1))
            _describe 'aliases' aliases
            ;;
    esac
}

_agent
"""

FISH_COMPLETION = """
function __agent_complete_aliases
    agent --list-aliases | grep -v "No aliases" | grep -v "Available" | sed 's/^[[:space:]]*//g' | cut -d ':' -f 1
end

complete -c agent -l mode -d "Execution mode" -xa "manual autonomous"
complete -c agent -l debug -d "Enable debug logging"
complete -c agent -l max-plan-iterations -d "Maximum number of iterations for plan verification"
complete -c agent -l no-direct-execution -d "Disable direct execution of shell commands"
complete -c agent -l no-refine-queries -d "Disable query refinement via API"
complete -c agent -l show-log -d "Display the log after completion"
complete -c agent -l save-alias -d "Save the provided query as an alias" -x
complete -c agent -l list-aliases -d "List all available command aliases"
complete -c agent -l use-alias -d "Run the query associated with the given alias name" -xa "(__agent_complete_aliases)"
complete -c agent -l delete-alias -d "Delete the specified alias" -xa "(__agent_complete_aliases)"
complete -c agent -l env-file -d "Path to the environment file" -r
"""

def print_completion_script(shell):
    """Print the completion script for the specified shell."""
    if shell == "bash":
        print(BASH_COMPLETION)
    elif shell == "zsh":
        print(ZSH_COMPLETION)
    elif shell == "fish":
        print(FISH_COMPLETION)
    else:
        print(f"Unsupported shell: {shell}", file=sys.stderr)
        sys.exit(1)

def install_completion(shell=None):
    """Install the completion script for the user's shell."""
    if shell is None:
        # Try to detect the current shell
        shell = os.path.basename(os.environ.get("SHELL", ""))
    
    home = Path.home()
    
    if shell == "bash":
        completion_file = home / ".bash_completion"
        script = BASH_COMPLETION
        
        # Create file if it doesn't exist
        if not completion_file.exists():
            with open(completion_file, "w") as f:
                f.write("# Bash completions\n\n")
        
        # Check if our completion is already installed
        with open(completion_file, "r") as f:
            if "_agent_completion" in f.read():
                print("Bash completion for agent is already installed.")
                return
        
        # Append our completion
        with open(completion_file, "a") as f:
            f.write("\n# Agent CLI completion\n")
            f.write(script)
        
        print(f"Bash completion installed to {completion_file}")
        print("Please restart your shell or run 'source ~/.bash_completion'")
        
    elif shell == "zsh":
        # For zsh, we'll create a completion file in a standard location
        completion_dir = home / ".zsh" / "completion"
        completion_dir.mkdir(parents=True, exist_ok=True)
        
        completion_file = completion_dir / "_agent"
        
        with open(completion_file, "w") as f:
            f.write(ZSH_COMPLETION)
        
        # Make sure the completion directory is in fpath
        zshrc = home / ".zshrc"
        zshrc_line = f'fpath=({completion_dir} $fpath)'
        
        if zshrc.exists():
            with open(zshrc, "r") as f:
                content = f.read()
            
            if str(completion_dir) not in content:
                with open(zshrc, "a") as f:
                    f.write(f"\n# Add agent completion\n{zshrc_line}\nautoload -Uz compinit && compinit\n")
        
        print(f"Zsh completion installed to {completion_file}")
        print("Please restart your shell or run 'source ~/.zshrc'")
        
    elif shell == "fish":
        # For fish, we'll create a completion file in the standard location
        completion_dir = home / ".config" / "fish" / "completions"
        completion_dir.mkdir(parents=True, exist_ok=True)
        
        completion_file = completion_dir / "agent.fish"
        
        with open(completion_file, "w") as f:
            f.write(FISH_COMPLETION)
        
        print(f"Fish completion installed to {completion_file}")
        print("Please restart your shell or run 'source ~/.config/fish/completions/agent.fish'")
        
    else:
        print(f"Unsupported shell: {shell}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: agent-completion [bash|zsh|fish] [--install]", file=sys.stderr)
        sys.exit(1)
    
    shell = sys.argv[1]
    
    if len(sys.argv) > 2 and sys.argv[2] == "--install":
        install_completion(shell)
    else:
        print_completion_script(shell)

if __name__ == "__main__":
    main() 