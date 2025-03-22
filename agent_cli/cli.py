#!/usr/bin/env python3
"""
Command-line interface for the AI Terminal agent.
This module provides the entry point for the 'agent' command.
"""

import sys
import os
import json
from pathlib import Path
import argparse
import shlex

# Add parent directory to sys.path to import ai_terminal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai_terminal import AITerminal, FileLogger

# Constants
ALIASES_FILE = Path.home() / ".ai_terminal_aliases.json"

def load_aliases():
    """Load command aliases from the aliases file."""
    if not ALIASES_FILE.exists():
        return {}
    
    try:
        with open(ALIASES_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_alias(name, query):
    """Save a command alias to the aliases file."""
    aliases = load_aliases()
    aliases[name] = query
    
    # Ensure the directory exists
    ALIASES_FILE.parent.mkdir(exist_ok=True)
    
    with open(ALIASES_FILE, 'w') as f:
        json.dump(aliases, f, indent=2)
    
    return True

def list_aliases():
    """List all available command aliases."""
    aliases = load_aliases()
    if not aliases:
        return "No aliases defined. Use 'agent --save-alias NAME \"QUERY\"' to create one."
    
    result = ["Available command aliases:"]
    for name, query in aliases.items():
        result.append(f"  {name}: \"{query}\"")
    
    return "\n".join(result)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="AI-Powered Terminal Agent")
    
    # Basic options
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query or task for the AI agent (in quotes)"
    )
    
    # Alias management
    alias_group = parser.add_argument_group('Alias Management')
    alias_group.add_argument(
        "--save-alias",
        metavar="NAME",
        help="Save the provided query as an alias with the given name"
    )
    alias_group.add_argument(
        "--list-aliases",
        action="store_true",
        help="List all available command aliases"
    )
    alias_group.add_argument(
        "--use-alias",
        metavar="NAME",
        help="Run the query associated with the given alias name"
    )
    alias_group.add_argument(
        "--delete-alias",
        metavar="NAME",
        help="Delete the specified alias"
    )
    
    # Advanced options
    parser.add_argument(
        "--mode",
        choices=["manual", "autonomous"],
        default=os.getenv("DEFAULT_MODE", "manual"),
        help="Execution mode - 'manual' or 'autonomous'"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to the environment file (default: .env)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--max-plan-iterations",
        type=int,
        default=3,
        help="Maximum number of iterations for plan verification"
    )
    parser.add_argument(
        "--no-direct-execution",
        action="store_true",
        help="Disable direct execution of shell commands"
    )
    parser.add_argument(
        "--no-refine-queries",
        action="store_true",
        help="Disable query refinement via API"
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Display the log after completion when running in one-shot mode"
    )
    
    # Override typical shell quoting behavior if needed
    if len(sys.argv) == 2 and " " in sys.argv[1] and not (sys.argv[1].startswith("'") or sys.argv[1].startswith('"')):
        # If a single argument with spaces is provided without quotes, 
        # treat the whole thing as the query
        args = parser.parse_args([sys.argv[1]])
    else:
        args = parser.parse_args()
    
    # Handle alias commands
    from rich.console import Console
    console = Console()
    
    if args.list_aliases:
        console.print(list_aliases())
        return
    
    if args.delete_alias:
        aliases = load_aliases()
        if args.delete_alias in aliases:
            del aliases[args.delete_alias]
            with open(ALIASES_FILE, 'w') as f:
                json.dump(aliases, f, indent=2)
            console.print(f"[green]Alias '{args.delete_alias}' deleted successfully.[/green]")
        else:
            console.print(f"[red]Alias '{args.delete_alias}' not found.[/red]")
        return
    
    if args.use_alias:
        aliases = load_aliases()
        if args.use_alias in aliases:
            args.query = aliases[args.use_alias]
            console.print(f"[green]Using alias '{args.use_alias}':[/green] {args.query}")
        else:
            console.print(f"[red]Alias '{args.use_alias}' not found.[/red]")
            return
    
    if args.save_alias:
        if not args.query:
            console.print("[red]Error: No query provided to save as alias.[/red]")
            return
        
        save_alias(args.save_alias, args.query)
        console.print(f"[green]Alias '{args.save_alias}' saved successfully for query:[/green] {args.query}")
        return
    
    # Load custom environment variables if specified
    if args.env_file and Path(args.env_file).exists():
        from dotenv import load_dotenv
        console.print(f"[dim]Loading environment from {args.env_file}[/dim]")
        load_dotenv(args.env_file, override=True)
    
    # Set default logs location
    logs_folder = Path.home() / ".ai_terminal_logs"
    max_log_files = 10  # Default number of log files to keep
    
    # Create a FileLogger
    custom_logger = FileLogger(
        logs_folder=logs_folder,
        max_log_files=max_log_files
    )
    
    # Initialize the AI Terminal
    terminal = AITerminal(
        mode=args.mode, 
        debug=args.debug, 
        max_plan_iterations=args.max_plan_iterations,
        direct_execution=not args.no_direct_execution,
        refine_queries=not args.no_refine_queries,
        logger=custom_logger
    )
    
    # Check if a query was provided
    if args.query:
        console.print(f"[bold green]AI Terminal Agent[/bold green]")
        console.print(f"[bold]Query:[/bold] {args.query}")
        console.print("=" * 50)
        
        # Process the request with one-shot mode enabled for more verbose output
        terminal.process_request(args.query, one_shot_mode=True)
        
        console.print("\n" + "=" * 50)
        console.print("[bold green]Request completed. Output saved to log.[/bold green]")
        console.print(f"[dim]Log file: {terminal.logger.log_file}[/dim]")
        
        if args.show_log:
            console.print("\n[bold blue]One-Shot Mode Log:[/bold blue]")
            console.print(terminal.logger.get_conversation_history())
    else:
        # Run in interactive mode
        terminal.run()


if __name__ == "__main__":
    main() 