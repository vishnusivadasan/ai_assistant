#!/usr/bin/env python3
"""
AI-Powered Terminal for macOS

This script creates an AI-powered terminal that interacts with the OpenAI API,
converts natural language instructions into shell commands, executes them, and
handles errors intelligently.
"""

import os
import sys
import re
import json
import sqlite3
import argparse
import subprocess
import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Confirm, Prompt
from rich.markdown import Markdown
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

# Initialize Rich console for pretty output
console = Console()

# Constants
DB_PATH = Path.home() / ".ai_terminal_memory.db"
MAX_MEMORY_ENTRIES = 50
DANGEROUS_COMMANDS = [
    "rm -rf /", 
    "rm -rf /*", 
    "dd if=/dev/zero", 
    ": () { : | : & }; :",
    "chmod -R 777 /",
    "mkfs",
    ":(){ :|:& };:",
    "> /dev/sda",
    "mv /home /dev/null"
]

class AITerminal:
    """AI-Powered Terminal application that converts natural language to shell commands."""
    
    def __init__(self, mode: str = "manual", debug: bool = False):
        """
        Initialize the AI Terminal.
        
        Args:
            mode: Execution mode - 'manual' or 'autonomous'
            debug: Enable debug logging
        """
        self.mode = mode
        self.debug = debug
        self.memory = MemoryManager()
        self.shell = os.environ.get("SHELL", "/bin/zsh")
        self.command_history = []
        self.setup_database()
        self.current_task = None
        
        # Check for API key
        if not openai.api_key:
            console.print("[bold red]Error: OPENAI_API_KEY not found in .env file[/bold red]")
            console.print("Please create a .env file with your OpenAI API key. See .env.example for format.")
            sys.exit(1)
            
        console.print(Panel.fit(
            "[bold blue]AI Terminal[/bold blue]\n"
            f"Mode: [bold]{'Autonomous' if self.mode == 'autonomous' else 'Manual'}[/bold]\n"
            "Type 'exit' or 'quit' to exit, 'help' for help.",
            title="Welcome"
        ))
    
    def setup_database(self):
        """Set up the SQLite database for memory storage."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            task TEXT,
            commands TEXT,
            results TEXT,
            success INTEGER
        )
        ''')
        
        # Create session table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS session (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            current_task TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_debug(self, message: str):
        """Log a debug message if debug mode is enabled."""
        if self.debug:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            console.print(f"[dim][{timestamp}] DEBUG: {message}[/dim]")
    
    def run(self):
        """Main loop for the AI Terminal."""
        try:
            while True:
                # Get user input
                user_input = Prompt.ask("\n[bold green]What would you like me to do?[/bold green]")
                
                # Check for exit commands
                if user_input.lower() in ("exit", "quit"):
                    console.print("[bold blue]Exiting AI Terminal. Goodbye![/bold blue]")
                    break
                
                # Check for help command
                if user_input.lower() == "help":
                    self.show_help()
                    continue
                    
                # Check for mode switch command
                if user_input.lower() in ("switch mode", "toggle mode"):
                    self.mode = "autonomous" if self.mode == "manual" else "manual"
                    console.print(f"[bold blue]Switched to {self.mode} mode[/bold blue]")
                    continue
                
                # Process the user's request
                self.process_request(user_input)
                
        except KeyboardInterrupt:
            console.print("\n[bold blue]Interrupted. Exiting AI Terminal. Goodbye![/bold blue]")
        except Exception as e:
            console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")
            if self.debug:
                import traceback
                console.print(traceback.format_exc())
    
    def show_help(self):
        """Show help information."""
        help_text = """
        # AI Terminal Help
        
        ## Basic Commands
        - `exit` or `quit`: Exit the AI Terminal
        - `help`: Show this help message
        - `switch mode` or `toggle mode`: Switch between manual and autonomous modes
        
        ## Current Mode
        - Current mode: {mode}
        
        ## Modes
        - **Manual Mode**: The AI suggests commands but will only ask for confirmation if they appear dangerous
        - **Autonomous Mode**: Same as manual mode but may perform more complex actions independently
        
        ## Safety Features
        - Commands are analyzed for potential risks using both rule-based checks and AI
        - You will only be asked for confirmation when a command is potentially dangerous
        - Non-dangerous commands execute automatically for better efficiency
        
        ## Examples
        - "Find all Python files in this directory"
        - "Install Node.js using Homebrew"
        - "Create a new React project in ~/Projects"
        - "Check disk usage and show the largest directories"
        """
        
        help_text = help_text.format(mode="Autonomous" if self.mode == "autonomous" else "Manual")
        console.print(Markdown(help_text))
    
    def process_request(self, user_input: str):
        """
        Process a user request.
        
        Args:
            user_input: Natural language request from the user
        """
        # Set the current task
        self.current_task = user_input
        self.memory.store_task(user_input)
        
        # If the task requires multiple steps, use agent mode
        if self.is_complex_task(user_input):
            console.print("[blue]This looks like a complex task. Using Agent Mode to break it down...[/blue]")
            self.agent_mode(user_input)
        else:
            # Generate a command for the request
            command = self.generate_command(user_input)
            self.execute_with_confirmation(command)
    
    def is_complex_task(self, user_input: str) -> bool:
        """
        Determine if a task is complex enough to warrant using Agent Mode.
        
        Args:
            user_input: User's natural language request
            
        Returns:
            bool: True if the task appears complex, False otherwise
        """
        # Use OpenAI to determine if this is a complex task requiring multiple steps
        messages = [
            {"role": "system", "content": "You are a helpful assistant that determines if a task requires multiple shell commands to complete. Respond with 'yes' or 'no'."},
            {"role": "user", "content": f"Does this task require multiple shell commands to complete? Task: {user_input}"}
        ]
        
        self.log_debug(f"Checking complexity of task: {user_input}")
        
        try:
            response = openai.chat.completions.create(
                model=openai_model,
                messages=messages,
                max_tokens=5,
                temperature=0
            )
            response_text = response.choices[0].message.content.strip().lower()
            self.log_debug(f"Complexity check response: {response_text}")
            return "yes" in response_text
        except Exception as e:
            self.log_debug(f"Error in complexity check: {str(e)}")
            # Default to simple task on error
            return False
    
    def generate_command(self, user_input: str, context: str = "") -> str:
        """
        Generate a shell command from natural language.
        
        Args:
            user_input: User's natural language request
            context: Optional context from previous commands or errors
            
        Returns:
            str: Generated shell command
        """
        system_message = """You are an AI assistant that converts natural language requests into macOS terminal commands.
        Generate the exact shell command(s) that would accomplish the task. Provide ONLY the command, nothing else.
        Use macOS-compatible syntax (zsh/bash). If multiple commands are needed, use && to chain them or ; for sequential execution.
        """
        
        user_message = user_input
        if context:
            user_message += f"\n\nContext from previous execution: {context}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        self.log_debug(f"Generating command for: {user_input}")
        
        try:
            response = openai.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=0.2
            )
            command = response.choices[0].message.content.strip()
            self.log_debug(f"Generated command: {command}")
            return command
        except Exception as e:
            console.print(f"[bold red]Error generating command: {str(e)}[/bold red]")
            return ""
    
    def execute_with_confirmation(self, command: str, context: str = ""):
        """
        Execute a command with user confirmation only if the command is potentially dangerous.
        
        Args:
            command: Shell command to execute
            context: Optional context about the command
        """
        if not command:
            console.print("[bold red]No valid command generated.[/bold red]")
            return
        
        # Check for dangerous commands using local rules and AI
        is_dangerous = self.is_dangerous_command(command)
        
        # Use AI to check if command is potentially dangerous when local check is negative
        if not is_dangerous:
            is_dangerous = self.check_command_danger_with_ai(command)
        
        # Show the command that will be executed
        console.print("\n[bold blue]Generated Command:[/bold blue]")
        console.print(Syntax(command, "bash", theme="monokai", line_numbers=False))
        
        if context:
            console.print(f"[dim]{context}[/dim]")
        
        # Only ask for confirmation if command is potentially dangerous
        execute = True
        if is_dangerous:
            console.print("[bold yellow]This command may have potential risks.[/bold yellow]")
            execute = Confirm.ask("Are you sure you want to execute this command?", default=False)
        else:
            # In either mode, run non-dangerous commands automatically
            console.print("[dim]Command appears safe, executing automatically...[/dim]")
        
        if execute:
            self.execute_command(command)
    
    def is_dangerous_command(self, command: str) -> bool:
        """
        Check if a command is potentially dangerous.
        
        Args:
            command: Shell command to check
            
        Returns:
            bool: True if command appears dangerous, False otherwise
        """
        command_lower = command.lower()
        
        # Check against list of known dangerous commands
        for dangerous_cmd in DANGEROUS_COMMANDS:
            if dangerous_cmd in command_lower:
                return True
        
        # Check for sudo commands that remove things
        if "sudo" in command_lower and ("rm -rf" in command_lower or "rmdir" in command_lower):
            return True
            
        # Check for commands that might affect large areas of the filesystem
        if "rm -rf" in command_lower and ("/" in command_lower or "~" in command_lower):
            # Allow if it's a specific subdirectory, but not broad areas
            if not re.search(r'rm -rf ["\']?[a-zA-Z0-9_\-\.]+["\']?/?$', command_lower):
                return True
        
        return False
    
    def check_command_danger_with_ai(self, command: str) -> bool:
        """
        Use AI to determine if a command is potentially dangerous.
        
        Args:
            command: Shell command to check
            
        Returns:
            bool: True if AI thinks command is dangerous, False otherwise
        """
        try:
            self.log_debug(f"Checking command danger with AI: {command}")
            
            # Create a system message for danger assessment
            system_message = """You are an AI assistant that assesses the safety of shell commands.
            Analyze the given command and determine if it's potentially dangerous or destructive.
            Consider risks like:
            - Data deletion or corruption
            - System modification that's hard to reverse
            - Security vulnerabilities
            - Resource exhaustion
            - Privilege escalation
            
            Respond with ONLY "dangerous" or "safe" without any explanation.
            """
            
            # Create the message list
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Assess this command: {command}"}
            ]
            
            # Get response from OpenAI
            response = openai.chat.completions.create(
                model=openai_model,
                messages=messages,
                max_tokens=10,
                temperature=0
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content.strip().lower()
            self.log_debug(f"AI danger assessment: {response_text}")
            
            # Check if the AI considers the command dangerous
            return "dangerous" in response_text
            
        except Exception as e:
            # Log the error
            self.log_debug(f"Error in AI danger check: {str(e)}")
            # Default to safe on API error to avoid disrupting workflow
            return False
    
    def execute_command(self, command: str) -> Tuple[str, bool]:
        """
        Execute a shell command and return its output.
        
        Args:
            command: Shell command to execute
            
        Returns:
            Tuple[str, bool]: Command output and success flag
        """
        try:
            # Add command to history
            self.command_history.append(command)
            
            # Create a process
            start_time = time.time()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                executable=self.shell,
                text=True
            )
            
            # Create a status indicator
            with console.status("[bold green]Executing command...[/bold green]") as status:
                # Poll for output while process is running
                output_lines = []
                error_lines = []
                
                while process.poll() is None:
                    # Read any new output
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        output_lines.append(stdout_line)
                        console.print(stdout_line, end="")
                    
                    # Read any new errors
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        error_lines.append(stderr_line)
                        console.print(f"[red]{stderr_line}[/red]", end="")
                    
                    # Avoid high CPU usage
                    time.sleep(0.01)
                
                # Get any remaining output
                stdout, stderr = process.communicate()
                if stdout:
                    output_lines.append(stdout)
                    console.print(stdout, end="")
                if stderr:
                    error_lines.append(stderr)
                    console.print(f"[red]{stderr}[/red]", end="")
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Combine output and errors
            output = "".join(output_lines)
            errors = "".join(error_lines)
            combined_output = output + errors
            
            # Determine success based on exit code
            success = process.returncode == 0
            status_str = "[bold green]Success[/bold green]" if success else "[bold red]Failed[/bold red]"
            console.print(f"\n{status_str} (Exit code: {process.returncode}, Time: {execution_time:.2f}s)")
            
            # Store command results in memory
            self.memory.store_command_result(
                self.current_task, 
                command, 
                combined_output, 
                success
            )
            
            # If the command failed, attempt to fix or provide context
            if not success and errors:
                self.handle_error(command, errors)
            
            return combined_output, success
            
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            return error_msg, False
    
    def handle_error(self, command: str, error_output: str):
        """
        Handle command execution errors.
        
        Args:
            command: Failed command
            error_output: Error message from command execution
        """
        console.print("[blue]Analyzing error...[/blue]")
        
        # Use OpenAI to analyze the error
        system_message = """You are an AI assistant that analyzes shell command errors and suggests fixes.
        Based on the error message, explain what went wrong and provide a corrected command.
        Format your response as JSON with these fields:
        - explanation: Brief explanation of what went wrong
        - fix_command: The corrected command to try (or null if no fix is suggested)
        - ask_user: Boolean indicating if user input is needed for the fix
        - user_prompt: Question to ask the user if ask_user is true (or null)
        """
        
        user_message = f"Command: {command}\n\nError output: {error_output}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = openai.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Display explanation
            console.print(f"[yellow]Error Analysis: {result['explanation']}[/yellow]")
            
            # If we need user input
            if result.get('ask_user', False) and result.get('user_prompt'):
                user_input = Prompt.ask(f"[bold]{result['user_prompt']}[/bold]")
                if user_input:
                    # Generate a new command with user input
                    new_context = f"Previous command '{command}' failed with error: {error_output}\nUser provided: {user_input}"
                    new_command = self.generate_command(self.current_task, new_context)
                    self.execute_with_confirmation(new_command, "Modified command based on error analysis")
            
            # If a fix is suggested
            elif result.get('fix_command'):
                console.print("[green]Suggested fix:[/green]")
                console.print(Syntax(result['fix_command'], "bash", theme="monokai", line_numbers=False))
                
                if self.mode == "autonomous" or Confirm.ask("Try this fix?", default=True):
                    self.execute_command(result['fix_command'])
            
            # If no fix is available
            else:
                console.print("[yellow]No automatic fix available for this error.[/yellow]")
                
        except Exception as e:
            console.print(f"[bold red]Error analyzing command failure: {str(e)}[/bold red]")
    
    def agent_mode(self, task: str):
        """
        Agent Mode for breaking down and executing complex tasks.
        
        Args:
            task: Complex task description
        """
        console.print("[bold blue]Agent Mode activated[/bold blue]")
        console.print(f"Task: [bold]{task}[/bold]")
        
        # Get step-by-step plan from OpenAI
        system_message = """You are an AI assistant that breaks down complex tasks into a series of shell commands for macOS.
        Create a step-by-step plan with specific shell commands to accomplish the user's goal.
        Format your response as JSON with an array of steps, each with:
        - description: Brief description of what this step does
        - command: The exact shell command to run
        - critical: Boolean indicating if this step is critical (failure should stop execution)
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Task: {task}"}
        ]
        
        try:
            response = openai.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            
            if 'steps' not in plan or not plan['steps']:
                console.print("[bold red]Failed to generate a plan for this task.[/bold red]")
                return
            
            # Display the plan
            console.print("\n[bold blue]Execution Plan:[/bold blue]")
            for i, step in enumerate(plan['steps'], 1):
                console.print(f"[bold]{i}.[/bold] {step['description']}")
                console.print(Syntax(step['command'], "bash", theme="monokai", line_numbers=False))
                console.print()
            
            # Ask for confirmation to execute the plan
            if self.mode == "manual" and not Confirm.ask("Execute this plan?", default=True):
                console.print("[blue]Plan execution cancelled.[/blue]")
                return
            
            # Execute each step in the plan
            for i, step in enumerate(plan['steps'], 1):
                console.print(f"\n[bold blue]Step {i}/{len(plan['steps'])}: {step['description']}[/bold blue]")
                
                # Execute with or without confirmation based on mode
                if self.mode == "manual":
                    self.execute_with_confirmation(step['command'], f"Step {i}/{len(plan['steps'])}")
                else:
                    console.print(Syntax(step['command'], "bash", theme="monokai", line_numbers=False))
                    output, success = self.execute_command(step['command'])
                    
                    # If a critical step failed, stop execution
                    if not success and step.get('critical', False):
                        console.print("[bold red]Critical step failed. Stopping execution.[/bold red]")
                        
                        # Ask if user wants to try to fix and continue
                        if Confirm.ask("Would you like me to try to fix this and continue?", default=True):
                            self.handle_error(step['command'], output)
                        else:
                            break
            
            console.print("\n[bold green]Plan execution completed.[/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]Error in Agent Mode: {str(e)}[/bold red]")
            if self.debug:
                import traceback
                console.print(traceback.format_exc())


class MemoryManager:
    """Manages the AI Terminal's memory and context awareness."""
    
    def __init__(self):
        """Initialize the Memory Manager."""
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def __del__(self):
        """Close database connection when object is destroyed."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
    
    def store_task(self, task: str):
        """
        Store a new task in the session.
        
        Args:
            task: User's task description
        """
        # Clear existing session data
        self.cursor.execute("DELETE FROM session")
        
        # Store new session
        timestamp = datetime.now().isoformat()
        self.cursor.execute(
            "INSERT INTO session (timestamp, current_task) VALUES (?, ?)",
            (timestamp, task)
        )
        self.conn.commit()
    
    def store_command_result(self, task: str, command: str, result: str, success: bool):
        """
        Store a command execution result in memory.
        
        Args:
            task: Associated task
            command: Executed command
            result: Command output
            success: Whether command succeeded
        """
        timestamp = datetime.now().isoformat()
        
        # Insert new memory entry
        self.cursor.execute(
            "INSERT INTO memory (timestamp, task, commands, results, success) VALUES (?, ?, ?, ?, ?)",
            (timestamp, task, command, result, 1 if success else 0)
        )
        
        # Prune old entries if we exceed the maximum
        self.cursor.execute("SELECT COUNT(*) FROM memory")
        count = self.cursor.fetchone()[0]
        
        if count > MAX_MEMORY_ENTRIES:
            # Delete oldest entries
            self.cursor.execute(
                "DELETE FROM memory WHERE id IN (SELECT id FROM memory ORDER BY timestamp ASC LIMIT ?)",
                (count - MAX_MEMORY_ENTRIES,)
            )
        
        self.conn.commit()
    
    def get_recent_memories(self, limit: int = 5) -> List[Dict]:
        """
        Get recent memory entries.
        
        Args:
            limit: Maximum number of entries to retrieve
            
        Returns:
            List[Dict]: Recent memory entries
        """
        self.cursor.execute(
            "SELECT * FROM memory ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        
        memories = []
        for row in self.cursor.fetchall():
            memories.append(dict(row))
        
        return memories
    
    def get_current_task(self) -> Optional[str]:
        """
        Get the current task from the session.
        
        Returns:
            Optional[str]: Current task or None
        """
        self.cursor.execute("SELECT current_task FROM session LIMIT 1")
        row = self.cursor.fetchone()
        
        if row:
            return row[0]
        return None
    
    def search_memories(self, query: str) -> List[Dict]:
        """
        Search memories for relevant entries.
        
        Args:
            query: Search terms
            
        Returns:
            List[Dict]: Matching memory entries
        """
        search_terms = f"%{query}%"
        self.cursor.execute(
            "SELECT * FROM memory WHERE task LIKE ? OR commands LIKE ? OR results LIKE ? ORDER BY timestamp DESC",
            (search_terms, search_terms, search_terms)
        )
        
        memories = []
        for row in self.cursor.fetchall():
            memories.append(dict(row))
        
        return memories


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AI-Powered Terminal")
    parser.add_argument(
        "--mode",
        choices=["manual", "autonomous"],
        default=os.getenv("DEFAULT_MODE", "manual"),
        help="Execution mode - 'manual' or 'autonomous'"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Initialize and run the AI Terminal
    terminal = AITerminal(mode=args.mode, debug=args.debug)
    terminal.run() 