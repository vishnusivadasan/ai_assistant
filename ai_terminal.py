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
import shlex
import glob
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
from utils.file_logger import FileLogger
from utils.memory_manager import MemoryManager

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
LOGS_FOLDER = Path.home() / ".ai_terminal_logs"
MAX_LOG_FILES = 10  # Maximum number of log files to keep
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
    
    def __init__(self, mode: str = "manual", debug: bool = False, max_plan_iterations: int = 5, 
                 direct_execution: bool = True, refine_queries: bool = True, logger=None):
        """
        Initialize the AI Terminal.
        
        Args:
            mode: Execution mode - 'manual' or 'autonomous'
            debug: Enable debug logging
            max_plan_iterations: Maximum number of iterations for plan verification
            direct_execution: Enable direct execution of shell commands if detected
            refine_queries: Enable query refinement via API
            logger: FileLogger instance, if None a default one will be created
        """
        self.mode = mode
        self.debug = debug
        self.memory = MemoryManager()  # Keep for backward compatibility
        self.memory.ai_terminal = self  # Set reference to this instance
        self.shell = os.environ.get("SHELL", "/bin/zsh")
        self.command_history = []
        self.max_plan_iterations = max_plan_iterations
        self.direct_execution = direct_execution
        self.refine_queries = refine_queries
        self.current_task = None
        
        # Set up logger
        if logger is None:
            self.logger = FileLogger()  # Create default file-based logger
            print("Created default logger")
        else:
            self.logger = logger
            print(f"Using provided logger with logs folder: {self.logger.logs_folder}")
        
        # Check for API key
        if not openai.api_key:
            console.print("[bold red]Error: OPENAI_API_KEY not found in .env file[/bold red]")
            console.print("Please create a .env file with your OpenAI API key. See .env.example for format.")
            sys.exit(1)
            
        console.print(Panel.fit(
            "[bold blue]AI Terminal[/bold blue]\n"
            f"Mode: [bold]{'Autonomous' if self.mode == 'autonomous' else 'Manual'}[/bold]\n"
            f"Max Plan Iterations: [bold]{self.max_plan_iterations}[/bold]\n"
            f"Direct Execution: [bold]{'Enabled' if self.direct_execution else 'Disabled'}[/bold]\n"
            f"Query Refinement: [bold]{'Enabled' if self.refine_queries else 'Disabled'}[/bold]\n"
            f"Log File: [bold]{self.logger.log_file}[/bold]\n"
            f"Logs Directory: [bold]{self.logger.logs_folder}[/bold]\n"
            "Type 'exit' or 'quit' to exit, 'help' for help.",
            title="Welcome"
        ))
        
        # Log system startup
        self.logger.log_system_message(f"AI Terminal started in {mode} mode with the following settings:\n"
                                       f"Max Plan Iterations: {max_plan_iterations}\n"
                                       f"Direct Execution: {'Enabled' if direct_execution else 'Disabled'}\n"
                                       f"Query Refinement: {'Enabled' if refine_queries else 'Disabled'}\n"
                                       f"Logs Directory: {self.logger.logs_folder}")
    
    def setup_database(self):
        """Set up the SQLite database for memory storage."""
        # This is now handled by the MemoryManager class
        pass
    
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
                    self.logger.log_system_message(f"Switched to {self.mode} mode")
                    continue
                
                # Check for log viewing commands
                if user_input.lower() in ("show log", "view log"):
                    self._show_current_log()
                    continue
                
                if user_input.lower() in ("show logs", "list logs"):
                    self._list_all_logs()
                    continue
                
                # Check for command to open logs directory
                if user_input.lower() in ("open logs", "open log folder", "open logs folder"):
                    self._open_logs_directory()
                    continue
                
                # Check for command to view last command output
                if user_input.lower() in ("show output", "view output", "show last output", "view last output"):
                    self._show_last_command_output()
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
    
    def _show_current_log(self):
        """Display the current conversation log."""
        log_content = self.logger.get_conversation_history()
        
        console.print("\n[bold blue]Current Conversation Log:[/bold blue]")
        console.print(Panel(log_content, title=f"Log File: {self.logger.log_file}", expand=False))
        
        self.logger.log_system_message("User viewed the current conversation log")
    
    def _list_all_logs(self):
        """List all available conversation logs."""
        log_files = sorted(glob.glob(str(self.logger.logs_folder / "conversation_*.log")), key=os.path.getmtime, reverse=True)
        
        console.print("\n[bold blue]Available Conversation Logs:[/bold blue]")
        
        for i, log_file in enumerate(log_files, 1):
            path = Path(log_file)
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            formatted_time = mtime.strftime("%Y-%m-%d %H:%M:%S")
            size_kb = path.stat().st_size / 1024
            
            console.print(f"{i}. [cyan]{path.name}[/cyan] - {formatted_time} ({size_kb:.1f} KB)")
        
        if not log_files:
            console.print("[yellow]No conversation logs found.[/yellow]")
        
        self.logger.log_system_message("User listed all conversation logs")
    
    def _open_logs_directory(self):
        """Open the logs directory in Finder."""
        console.print(f"\n[blue]Opening logs directory: {self.logger.logs_folder}[/blue]")
        
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(self.logger.logs_folder)])
            elif sys.platform == 'win32':  # Windows
                subprocess.run(['explorer', str(self.logger.logs_folder)])
            elif sys.platform == 'linux':  # Linux
                subprocess.run(['xdg-open', str(self.logger.logs_folder)])
            else:
                console.print(f"[yellow]Unsupported platform: {sys.platform}. Please manually navigate to {self.logger.logs_folder}[/yellow]")
            
            self.logger.log_system_message(f"User opened logs directory: {self.logger.logs_folder}")
        except Exception as e:
            console.print(f"[bold red]Error opening logs directory: {str(e)}[/bold red]")
            self.logger.log_system_message(f"Error opening logs directory: {str(e)}")
    
    def _show_last_command_output(self):
        """Display the full output of the last executed command."""
        try:
            # Read the log file to find the last command and result
            log_content = self.logger.get_conversation_history()
            
            # Split the log content into sections
            sections = log_content.split("\nCOMMAND [")
            
            if len(sections) > 1:
                # Get the last command section
                last_command_section = "\nCOMMAND [" + sections[-1]
                
                # Further split to get just the result
                result_parts = last_command_section.split("\nRESULT [")
                
                if len(result_parts) > 1:
                    result_section = "\nRESULT [" + result_parts[-1]
                    
                    # Extract the command and result content
                    command_content = result_parts[0].split("]: ", 1)[1].strip()
                    
                    # Get the result content (excluding the timestamp and status)
                    result_lines = result_section.split("\n")
                    status_line = result_lines[0]  # Contains timestamp and status
                    result_content = "\n".join(result_lines[1:])
                    
                    # Get status information
                    status = "SUCCESS" if "SUCCESS" in status_line else "FAILURE"
                    status_color = "green" if status == "SUCCESS" else "red"
                    
                    # Display in a nice panel with clear formatting
                    console.print("\n[bold blue]Complete Output of Last Command:[/bold blue]")
                    
                    # Command panel
                    console.print(Panel(
                        Syntax(command_content, "bash", theme="monokai", line_numbers=False),
                        title="[bold yellow]Command[/bold yellow]",
                        expand=False
                    ))
                    
                    # Status line
                    console.print(f"[bold {status_color}]{status}[/bold {status_color}]")
                    
                    # Output panel - use a larger panel for the output
                    console.print(Panel(
                        result_content,
                        title="[bold cyan]Complete Output[/bold cyan]",
                        expand=False,
                        width=min(console.width, 120)
                    ))
                    
                    self.logger.log_system_message("User viewed the full output of the last command")
                else:
                    console.print("[yellow]No command results found in the log.[/yellow]")
            else:
                console.print("[yellow]No previous commands found in the log.[/yellow]")
                
        except Exception as e:
            console.print(f"[bold red]Error retrieving last command output: {str(e)}[/bold red]")
            self.logger.log_system_message(f"Error retrieving last command output: {str(e)}")
    
    def show_help(self):
        """Show help information."""
        help_text = """
        # AI Terminal Help
        
        ## Basic Commands
        - `exit` or `quit`: Exit the AI Terminal
        - `help`: Show this help message
        - `switch mode` or `toggle mode`: Switch between manual and autonomous modes
        - `show log` or `view log`: Display the current conversation log
        - `show logs` or `list logs`: List all available conversation logs
        - `open logs` or `open logs folder`: Open the logs directory in your file explorer
        - `show output` or `view output`: Display the full output of the last executed command
        
        ## Current Mode
        - Current mode: {mode}
        - Max plan iterations: {max_iterations}
        - Direct execution: {direct_execution}
        - Query refinement: {query_refinement}
        - Log file: {log_file}
        - Logs directory: {logs_folder}
        
        ## Modes
        - **Manual Mode**: The AI suggests commands but will only ask for confirmation if they appear dangerous
        - **Autonomous Mode**: Same as manual mode but may perform more complex actions independently
        
        ## Features
        - **Direct Execution**: When enabled, shell commands entered directly are executed without AI processing
        - **Query Refinement**: When enabled, user inputs are refined via API for better clarity
        - **Plan Verification**: Complex tasks go through plan verification and refinement (up to {max_iterations} iterations)
        - **Conversation Logging**: All interactions are logged to plain text files in {logs_folder}
        - **Full Output Capture**: Complete command output is captured and can be viewed with 'show output'
        - **Content Creation**: Can generate reports, articles, and other content with proper formatting
        - **General Information**: Can answer general knowledge questions without executing commands
        
        ## Context Awareness
        The AI Terminal remembers your previous commands, their results, and conversations.
        You can ask questions about previous operations, for example:
        - "What was the last command you executed?"
        - "What was the output of the last command?"
        - "Show me what you've done so far"
        - "What command did you use to do X?"
        
        ## Content Creation Capabilities
        AI Terminal can now create content like reports, articles, and documentation.
        For example, you can ask:
        - "Write a report on the history of India from 2010 to 2020"
        - "Create a detailed tutorial on how to use Docker"
        - "Generate a summary of recent AI advancements"
        
        The system will:
        1. Generate a structured outline
        2. Let you approve the outline
        3. Create comprehensive content
        4. Allow you to save the content as a Markdown file
        
        ## General Information Questions
        You can ask general knowledge questions directly, and the AI will answer them without executing commands:
        - "What is the capital of France?"
        - "When is Christmas?"
        - "How many days are in a year?"
        - "Who was the first person to walk on the moon?"
        - "Explain how photosynthesis works"
        
        ## Safety Features
        - Commands are analyzed for potential risks using both rule-based checks and AI
        - You will only be asked for confirmation when a command is potentially dangerous
        - Non-dangerous commands execute automatically for better efficiency
        
        ## Examples
        - "Find all Python files in this directory"
        - "Install Node.js using Homebrew"
        - "Create a new React project in ~/Projects"
        - "Check disk usage and show the largest directories"
        - "Write a detailed report on renewable energy sources"
        - "ls -la" (direct execution)
        - "What was the last command you ran?" (context query)
        - "What is the capital of Japan?" (general information question)
        """
        
        help_text = help_text.format(
            mode="Autonomous" if self.mode == "autonomous" else "Manual",
            max_iterations=self.max_plan_iterations,
            direct_execution="Enabled" if self.direct_execution else "Disabled",
            query_refinement="Enabled" if self.refine_queries else "Disabled",
            log_file=self.logger.log_file,
            logs_folder=self.logger.logs_folder
        )
        console.print(Markdown(help_text))
    
    def is_general_information_question(self, user_input: str) -> bool:
        """
        Determine if the user input is a general information question using our API-based classification.
        This method is kept for backward compatibility but now uses the unified API-based approach.
        
        Args:
            user_input: User's input text
            
        Returns:
            bool: True if the input is classified as a general information question, False otherwise
        """
        self.log_debug(f"Checking if query is a general information question: {user_input}")
        
        # Use the API-based classification instead of a separate API call
        intent = self.classify_user_intent(user_input)
        
        # Log the classification result
        self.log_debug(f"General information check via API: {intent['primary_intent']}")
        
        # Return true if the API classified this as a general information question
        return intent['primary_intent'] == 'general_information'
    
    def handle_general_information_question(self, question: str):
        """
        Handle general information questions by using the AI to generate a direct answer
        without executing any commands.
        
        Args:
            question: User's question
        """
        console.print("[blue]This appears to be a general information question. Let me answer that for you...[/blue]")
        self.logger.log_system_message("API classified this as a general information question, providing direct answer")
        
        # Create a system message for general information responses
        system_message = """You are a helpful AI assistant answering general information questions.
        Provide concise, accurate answers to the user's question based on your knowledge.
        If the question is ambiguous, ask for clarification.
        If you're not confident in the accuracy of your answer, acknowledge the uncertainty.
        Respond in a conversational, helpful manner.
        Format your response using Markdown for readability.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        
        try:
            # Use the animation helper for the API call
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.3
            )
            
            # Display the response
            answer = response.choices[0].message.content.strip()
            console.print(Markdown(answer))
            
            # Log the response metadata
            self.logger.log_system_message(f"Response to general information question: {question}")
            # Log the complete answer content using the dedicated method
            self.logger.log_answer(answer)
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(f"Error: {error_msg}")
    
    def process_request(self, user_input: str, one_shot_mode: bool = False):
        """
        Process a user request.
        
        Args:
            user_input: Natural language request from the user
            one_shot_mode: Whether executing in one-shot mode (for more verbose output)
        """
        # Set the current task
        self.current_task = user_input
        
        # Log the user query
        self.logger.log_system_message(f"Processing request: {user_input}")
        self.logger.log_user_query(user_input)
        
        if one_shot_mode:
            console.print(f"[bold blue]Processing request in {self.mode} mode[/bold blue]")
            console.print(f"[dim]Direct execution: {'Enabled' if self.direct_execution else 'Disabled'}[/dim]")
            console.print(f"[dim]Query refinement: {'Enabled' if self.refine_queries else 'Disabled'}[/dim]")
        
        # Refine the query first if enabled
        if self.refine_queries:
            refined_input = self.refine_user_query(user_input)
            if refined_input != user_input:
                if one_shot_mode:
                    console.print(f"[bold cyan]Original query:[/bold cyan] {user_input}")
                    console.print(f"[bold cyan]Refined query:[/bold cyan] {refined_input}")
                
                self.current_task = refined_input
                self.logger.log_system_message(f"Refined query: {refined_input}")
            else:
                refined_input = user_input
        else:
            refined_input = user_input
            
        # Classify user intent using API-based approach on the refined input
        intent = self.classify_user_intent(refined_input)
        primary_intent = intent['primary_intent']
        confidence = intent['confidence']
        
        # Log the classification result if in verbose mode
        if one_shot_mode or self.debug:
            console.print(f"[dim]Intent classification: {primary_intent} (confidence: {confidence:.2f})[/dim]")
            if 'analysis' in intent:
                console.print(f"[dim]Analysis: {intent['analysis']}[/dim]")
        
        # Handle different intents based on the classification
        if primary_intent == "general_information":
            # Handle general information questions
            self.handle_general_information_question(refined_input)
            return
        
        elif primary_intent == "content_creation":
            # Handle content creation tasks
            console.print("[blue]Content creation task detected. Entering Agent Mode...[/blue]")
            self.logger.log_system_message("Content creation task detected - entering Agent Mode directly")
            self.agent_mode(refined_input)
            return
        
        elif primary_intent == "context_request":
            # Handle context-related questions
            self.handle_context_request(refined_input)
            return
        
        elif primary_intent == "direct_command" and self.direct_execution:
            # Handle direct shell commands
            console.print("[blue]Detected shell command. Executing directly...[/blue]")
            
            # Get danger assessment from intent classification or perform it
            is_dangerous = intent.get('dangerous_command', False)
            if 'dangerous_command' not in intent:
                is_dangerous = self.check_command_danger_with_ai(refined_input)
                
            if is_dangerous:
                console.print("[bold yellow]This command may have potential risks.[/bold yellow]")
                self.execute_with_confirmation(refined_input, "Direct shell command execution", is_dangerous=True)
            else:
                self.execute_command(refined_input)
            return
        
        # Check if this is a complex task requiring agent mode
        if primary_intent == "complex_task" or intent.get('requires_agent_mode', False):
            console.print("[blue]This looks like a complex task. Using Agent Mode to break it down...[/blue]")
            self.agent_mode(refined_input)
        else:
            # Handle as a simple task requiring a single command
            command = self.generate_command(refined_input)
            
            # Check if the command is dangerous using the API classification
            is_dangerous = intent.get('dangerous_command', False)
            
            # If classification didn't include danger assessment, double-check
            if 'dangerous_command' not in intent:
                is_dangerous = self.check_command_danger_with_ai(command)
                
            self.execute_with_confirmation(command, is_dangerous=is_dangerous)
    
    def is_context_request(self, user_input: str) -> bool:
        """
        Determine if the user is asking about previous actions or context.
        This method is kept for backward compatibility but now uses the API-based approach.
        
        Args:
            user_input: User's input
            
        Returns:
            bool: True if this is a context-related question
        """
        # Use the API-based classification instead of rule-based patterns
        intent = self.classify_user_intent(user_input)
        
        # Log the classification result
        self.log_debug(f"Context request check via API: {intent['primary_intent']}")
        
        # Return true if the API classified this as a context request
        return intent['primary_intent'] == 'context_request'
    
    def is_complex_task(self, user_input: str) -> bool:
        """
        Determine if a task is complex enough to warrant using Agent Mode.
        This method is kept for backward compatibility but now uses the API-based approach.
        
        Args:
            user_input: User's natural language request
            
        Returns:
            bool: True if the task appears complex, False otherwise
        """
        # Use the API-based classification instead of rule-based patterns
        intent = self.classify_user_intent(user_input)
        
        # Log the classification result
        self.log_debug(f"Complex task check via API: {intent['primary_intent']}")
        
        # Check if the API classified this as a complex task or requiring agent mode
        return intent['primary_intent'] == 'complex_task' or intent.get('requires_agent_mode', False)
    
    def handle_context_request(self, user_input: str):
        """
        Handle requests about previous actions or context.
        
        Args:
            user_input: User's question about context
        """
        console.print("[blue]Searching conversation history for context...[/blue]")
        
        # Get conversation history from log file
        conversation_history = self.logger.get_conversation_history()
        
        # Use OpenAI to generate a response based on the context
        system_message = """You are an AI assistant that helps users understand their command-line history and context.
        Answer the user's question about previous commands, files, or actions based on the provided conversation history.
        Be specific, accurate, and brief in your response.
        If you can't find relevant information in the history, politely say so.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Question: {user_input}\n\nConversation History:\n{conversation_history}"}
        ]
        
        try:
            # Use the animation helper for the API call
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.3
            )
            
            # Display the response
            answer = response.choices[0].message.content.strip()
            console.print(Markdown(answer))
            
            # Log the response
            self.logger.log_system_message(f"Response to context query: {answer}")
            
        except Exception as e:
            error_msg = f"Error generating context response: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(f"Error: {error_msg}")
            
    def generate_command(self, user_input: str, context: str = "") -> str:
        """
        Generate a shell command from natural language.
        
        Args:
            user_input: User's natural language request
            context: Optional context from previous commands or errors
            
        Returns:
            str: Generated shell command
        """
        # Check if this is a content creation task using API-based classification
        intent = self.classify_user_intent(user_input)
        if intent['primary_intent'] == 'content_creation':
            self.log_debug(f"Content creation task detected in generate_command via API: {user_input}")
            return "echo 'This is a content creation task that will be handled by Agent Mode. Please try again and make sure direct execution is not enabled.'"
        
        system_message = """You are an AI assistant that converts natural language requests into macOS terminal commands.
        Generate the exact shell command(s) that would accomplish the task. Provide ONLY the command, nothing else.
        Use macOS-compatible syntax (zsh/bash). If multiple commands are needed, use && to chain them or ; for sequential execution.
        Do not wrap your response in code blocks or markdown formatting.
        """
        
        # Get directory context
        dir_context = self.get_directory_context()
        
        # Get conversation history from log file
        conversation_history = self.logger.get_conversation_history()
        
        user_message = user_input
        if context:
            user_message += f"\n\nContext from previous execution: {context}"
        
        # Add directory and conversation context
        user_message += f"\n\nDirectory Information:\n{dir_context}"
        user_message += f"\n\nRecent Conversation History:\n{conversation_history}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        self.log_debug(f"Generating command for: {user_input}")
        
        try:
            # Show thinking animation while waiting for API response
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.2
            )
            command = response.choices[0].message.content.strip()
            
            # Clean up the command by removing Markdown code block markers if present
            # Remove ```bash or ``` markers
            command = re.sub(r'^```(?:bash|sh)?\s*', '', command)
            command = re.sub(r'\s*```$', '', command)
            
            self.log_debug(f"Generated command (after cleanup): {command}")
            
            # Log the generated command
            self.logger.log_command(command)
            
            return command
        except Exception as e:
            error_msg = f"Error generating command: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(f"Error: {error_msg}")
            return ""
    
    def execute_with_confirmation(self, command: str, context: str = "", is_dangerous: bool = False) -> Tuple[str, bool]:
        """
        Execute a command with user confirmation only if the command is potentially dangerous.
        
        Args:
            command: Shell command to execute
            context: Optional context about the command
            is_dangerous: Whether the command is potentially dangerous
            
        Returns:
            Tuple[str, bool]: The command output and a boolean indicating success
        """
        if not command:
            console.print("[bold red]No valid command generated.[/bold red]")
            return "", False
        
        # Show the command that will be executed
        console.print("\n[bold blue]Generated Command:[/bold blue]")
        console.print(Syntax(command, "bash", theme="monokai", line_numbers=False))
        
        if context:
            console.print(f"[dim]{context}[/dim]")
        
        # If danger wasn't already assessed, use API-based check
        if not is_dangerous:
            is_dangerous = self.check_command_danger_with_ai(command)
            
        # Only ask for confirmation if command is potentially dangerous
        execute = True
        if is_dangerous:
            console.print("[bold yellow]This command may have potential risks.[/bold yellow]")
            execute = Confirm.ask("Are you sure you want to execute this command?", default=False)
        else:
            # In either mode, run non-dangerous commands automatically
            console.print("[dim]Command appears safe, executing automatically...[/dim]")
        
        if execute:
            output, success = self.execute_command(command)
            return output, success
        else:
            return "Command execution cancelled by user.", False
    
    def is_dangerous_command(self, command: str) -> bool:
        """
        Check if a command is potentially dangerous.
        This method is kept for backward compatibility but now uses the API-based approach.
        
        Args:
            command: Shell command to check
            
        Returns:
            bool: True if command appears dangerous, False otherwise
        """
        self.log_debug(f"is_dangerous_command called (using API-based approach): {command}")
        return self.check_command_danger_with_ai(command)
    
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
            
            # First try to use the classification result if it already includes danger assessment
            intent = self.classify_user_intent(command)
            if 'dangerous_command' in intent:
                danger_assessment = intent['dangerous_command']
                self.log_debug(f"Using danger assessment from classify_user_intent: {danger_assessment}")
                
                # Log the assessment details
                if 'risk_level' in intent:
                    self.logger.log_system_message(f"Command risk level: {intent['risk_level']}/10")
                if 'risk_reasons' in intent and intent['risk_reasons']:
                    if isinstance(intent['risk_reasons'], list):
                        self.logger.log_system_message(f"Risk reasons: {', '.join(intent['risk_reasons'])}")
                    else:
                        self.logger.log_system_message(f"Risk reason: {intent['risk_reasons']}")
                        
                return danger_assessment
            
            # If classification didn't include danger assessment, use a dedicated check
            # Get directory context
            dir_context = self.get_directory_context()
            
            # Create a system message for danger assessment
            system_message = """You are an AI assistant that assesses the safety of shell commands.
            Analyze the given command and determine if it's potentially dangerous or destructive.
            Consider risks like:
            - Data deletion or corruption
            - System modification that's hard to reverse
            - Security vulnerabilities
            - Resource exhaustion
            - Privilege escalation
            - Execution of untrusted code
            - Files being moved to unintended locations
            - Important system files being modified
            
            Respond with a JSON object containing:
            - "is_dangerous": true/false indicating if the command is potentially risky
            - "reason": Brief explanation of your assessment
            - "risk_level": A number from 0-10 indicating the risk level (0 = no risk, 10 = extreme risk)
            - "risk_areas": List of specific areas of risk (e.g., "data loss", "system integrity", etc.)
            
            Be cautious - if there's any significant risk, mark the command as dangerous.
            """
            
            # Create the message list
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Assess this command: {command}\n\nDirectory Information:\n{dir_context}"}
            ]
            
            # Get response from OpenAI with animation
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            self.log_debug(f"AI danger assessment: {result}")
            
            # Log the assessment
            self.logger.log_system_message(f"Command danger assessment: {'Dangerous' if result['is_dangerous'] else 'Safe'} (Risk level: {result['risk_level']})")
            if 'reason' in result:
                self.logger.log_system_message(f"Reason: {result['reason']}")
            if 'risk_areas' in result and result['risk_areas']:
                if isinstance(result['risk_areas'], list):
                    self.logger.log_system_message(f"Risk areas: {', '.join(result['risk_areas'])}")
                else:
                    self.logger.log_system_message(f"Risk area: {result['risk_areas']}")
            
            # Return the danger assessment
            return result['is_dangerous']
            
        except Exception as e:
            # Log the error
            self.log_debug(f"Error in AI danger check: {str(e)}")
            self.logger.log_system_message(f"Error in danger assessment: {str(e)}")
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
            
            # First try to determine if this is a simple command that will have short output
            # If so, use subprocess.run which is more reliable for capturing all output
            is_complex_command = ('|' in command or '>' in command or 
                                 any(cmd in command for cmd in ['find', 'grep', 'awk', 'sort']))
            is_interactive = any(cmd in command for cmd in ['vim', 'nano', 'less', 'more', 'top'])
            
            # Start timing
            start_time = time.time()
            
            if not is_complex_command and not is_interactive:
                # Use subprocess.run for simpler commands
                with console.status("[bold green]Executing command...[/bold green]"):
                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=True,
                        executable=self.shell,
                        text=True
                    )
                
                output = result.stdout
                errors = result.stderr
                success = result.returncode == 0
                
                # Print a separator line to make the result more visible
                console.print("\n" + "-" * 50)
                
                # Display captured output
                if output.strip():
                    console.print("\n[bold cyan]Command Output:[/bold cyan]")
                    console.print(output)
                    
                if errors.strip():
                    console.print("\n[bold red]Command Errors:[/bold red]")
                    console.print(errors)
                
                # Print a separator line to make the end of output clear
                console.print("-" * 50)
            
            else:
                # Use interactive approach for complex commands
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                    executable=self.shell,
                    text=True,
                    bufsize=1  # Line buffered
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
                            console.print(stdout_line, end="", highlight=False)
                        
                        # Read any new errors
                        stderr_line = process.stderr.readline()
                        if stderr_line:
                            error_lines.append(stderr_line)
                            console.print(f"[red]{stderr_line}[/red]", end="", highlight=False)
                        
                        # Avoid high CPU usage
                        time.sleep(0.01)
                    
                    # Get any remaining output
                    stdout, stderr = process.communicate()
                    if stdout:
                        output_lines.append(stdout)
                        console.print(stdout, end="", highlight=False)
                    if stderr:
                        error_lines.append(stderr)
                        console.print(f"[red]{stderr}[/red]", end="", highlight=False)
                
                # Combine output and errors
                output = "".join(output_lines)
                errors = "".join(error_lines)
                success = process.returncode == 0
                
                # Print separator and output summary for complex commands
                console.print("\n" + "-" * 50)
                console.print("[bold cyan]Output shown above - captured and logged[/bold cyan]")
                console.print("-" * 50)
            
            combined_output = output + errors
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Show execution status
            status_str = "[bold green]Success[/bold green]" if success else "[bold red]Failed[/bold red]"
            console.print(f"\n{status_str} (Exit code: {0 if success else 1}, Time: {execution_time:.2f}s)")
            console.print(f"[dim]Full output captured and logged to: {self.logger.log_file}[/dim]")
            
            # Log the result
            self.logger.log_result(combined_output, success)
            
            # If the command failed, attempt to fix or provide context
            if not success and errors:
                self.handle_error(command, errors)
            
            return combined_output, success
            
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(f"Error: {error_msg}")
            return error_msg, False
    
    def handle_error(self, command: str, error_output: str):
        """
        Handle command execution errors.
        
        Args:
            command: Failed command
            error_output: Error message from command execution
        """
        console.print("[blue]Analyzing error...[/blue]")
        self.logger.log_system_message("Analyzing command error...")
        
        # Use OpenAI to analyze the error
        system_message = """You are an AI assistant that analyzes shell command errors and suggests fixes.
        Based on the error message, explain what went wrong and provide a corrected command.
        Format your response as JSON with these fields:
        - explanation: Brief explanation of what went wrong
        - fix_command: The corrected command to try (or null if no fix is suggested)
        - ask_user: Boolean indicating if user input is needed for the fix
        - user_prompt: Question to ask the user if ask_user is true (or null)
        """
        
        # Get directory context
        dir_context = self.get_directory_context()
        
        # Get conversation history
        conv_history = self.logger.get_conversation_history()
        
        user_message = f"Command: {command}\n\nError output: {error_output}\n\nDirectory Information:\n{dir_context}\n\nRecent Conversation History:\n{conv_history}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            # Show thinking animation while waiting for API response
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Display explanation
            console.print(f"[yellow]Error Analysis: {result['explanation']}[/yellow]")
            self.logger.log_system_message(f"Error Analysis: {result['explanation']}")
            
            # If we need user input
            if result.get('ask_user', False) and result.get('user_prompt'):
                user_input = Prompt.ask(f"[bold]{result['user_prompt']}[/bold]")
                self.logger.log_user_query(f"Response to prompt: {user_input}")
                
                if user_input:
                    # Generate a new command with user input
                    new_context = f"Previous command '{command}' failed with error: {error_output}\nUser provided: {user_input}"
                    new_command = self.generate_command(self.current_task, new_context)
                    self.execute_with_confirmation(new_command, "Modified command based on error analysis")
            
            # If a fix is suggested
            elif result.get('fix_command'):
                console.print("[green]Suggested fix:[/green]")
                console.print(Syntax(result['fix_command'], "bash", theme="monokai", line_numbers=False))
                self.logger.log_system_message(f"Suggested fix: {result['fix_command']}")
                
                if self.mode == "autonomous" or Confirm.ask("Try this fix?", default=True):
                    self.execute_command(result['fix_command'])
            
            # If no fix is available
            else:
                console.print("[yellow]No automatic fix available for this error.[/yellow]")
                self.logger.log_system_message("No automatic fix available for this error.")
                
        except Exception as e:
            error_msg = f"Error analyzing command failure: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(f"Error: {error_msg}")
    
    def agent_mode(self, task: str):
        """
        Agent Mode for breaking down and executing complex tasks.
        
        Args:
            task: Complex task description
        """
        console.print("[bold blue]Agent Mode activated[/bold blue]")
        console.print(f"Task: [bold]{task}[/bold]")
        
        # Use API for complete task classification and analysis
        intent = self.classify_user_intent(task)
        primary_intent = intent.get('primary_intent', 'complex_task')
        
        # Log the classification result
        self.logger.log_system_message(f"Agent mode task classification: {primary_intent}")
        if 'analysis' in intent:
            self.logger.log_system_message(f"Task analysis: {intent['analysis']}")
        
        # Handle content creation tasks differently
        if primary_intent == 'content_creation':
            console.print("[bold blue]Content creation task detected. Generating content...[/bold blue]")
            self.handle_content_creation_task(task)
            return
        
        # Get initial high-level plan (without specific commands)
        high_level_plan = self.generate_high_level_plan(task)
        
        if not high_level_plan or 'steps' not in high_level_plan or not high_level_plan['steps']:
            console.print("[bold red]Failed to generate a plan for this task.[/bold red]")
            return
        
        # Group steps by phase
        analysis_steps = [s for s in high_level_plan['steps'] if s.get('phase', '') == 'analysis']
        planning_steps = [s for s in high_level_plan['steps'] if s.get('phase', '') == 'planning']
        execution_steps = [s for s in high_level_plan['steps'] if s.get('phase', '') == 'execution']
        other_steps = [s for s in high_level_plan['steps'] if s.get('phase', '') not in ['analysis', 'planning', 'execution']]
        
        # Display the high-level plan with phases
        console.print("\n[bold blue]High-Level Execution Plan:[/bold blue]")
        
        # Display phases separately with headers
        if analysis_steps:
            console.print("\n[bold yellow]PHASE 1: ANALYSIS & DISCOVERY[/bold yellow]")
            for i, step in enumerate(analysis_steps, 1):
                console.print(f"[bold]{i}.[/bold] {step['description']}")
        
        if planning_steps:
            console.print("\n[bold green]PHASE 2: PLANNING[/bold green]")
            for i, step in enumerate(planning_steps, 1):
                console.print(f"[bold]{i}.[/bold] {step['description']}")
        
        if execution_steps:
            console.print("\n[bold blue]PHASE 3: EXECUTION[/bold blue]")
            for i, step in enumerate(execution_steps, 1):
                console.print(f"[bold]{i}.[/bold] {step['description']}")
        
        if other_steps:
            console.print("\n[bold purple]OTHER STEPS[/bold purple]")
            for i, step in enumerate(other_steps, 1):
                console.print(f"[bold]{i}.[/bold] {step['description']}")
        
        # Ask for confirmation to execute the plan
        if self.mode == "manual" and not Confirm.ask("Execute this plan?", default=True):
            console.print("[blue]Plan execution cancelled.[/blue]")
            return
        
        # Track execution results for summary
        executed_steps = []
        successful_steps = []
        failed_steps = []
        early_termination = False
        
        # Execute each step in the plan - retain original order for execution
        current_phase = None
        
        for i, step in enumerate(high_level_plan['steps'], 1):
            # Check if we're entering a new phase
            step_phase = step.get('phase', 'other')
            if step_phase != current_phase:
                current_phase = step_phase
                phase_name = {
                    'analysis': "[bold yellow]PHASE 1: ANALYSIS & DISCOVERY[/bold yellow]",
                    'planning': "[bold green]PHASE 2: PLANNING[/bold green]",
                    'execution': "[bold blue]PHASE 3: EXECUTION[/bold blue]",
                    'other': "[bold purple]OTHER STEPS[/bold purple]"
                }.get(step_phase, "[bold purple]OTHER STEPS[/bold purple]")
                console.print(f"\n{phase_name}")
            
            console.print(f"\n[bold blue]Step {i}/{len(high_level_plan['steps'])}: {step['description']}[/bold blue]")
            
            # Generate the command for this specific step on the fly
            command = self.generate_step_command(task, step, high_level_plan['steps'][:i-1])
            
            # Display and execute the command
            console.print(Syntax(command, "bash", theme="monokai", line_numbers=False))
            
            # Check if the command is dangerous
            danger_assessment = self.check_command_danger_with_ai(command)
            
            # Execute with or without confirmation based on mode and danger level
            if self.mode == "manual" or danger_assessment:
                output, success = self.execute_with_confirmation(command, f"Step {i}/{len(high_level_plan['steps'])}", is_dangerous=danger_assessment)
            else:
                output, success = self.execute_command(command)
            
            # Verify if the step achieved its intended purpose
            step_verification = self.verify_step_success(step, command, output, success)
            
            # Track execution results for summary
            step_result = {
                "step_number": i,
                "description": step['description'],
                "command": command,
                "output": output,
                "success": step_verification['success'],
                "message": step_verification['message'],
                "phase": step_phase,
                "expected_outcome": step.get('expected_outcome', 'No expected outcome provided')
            }
            executed_steps.append(step_result)
            
            if step_verification['success']:
                successful_steps.append(step_result)
                console.print(f"[bold green]Step verification: {step_verification['message']}[/bold green]")
            else:
                failed_steps.append(step_result)
                console.print(f"[bold yellow]Step verification: {step_verification['message']}[/bold yellow]")
                
                # If a critical step failed or verification failed, ask to fix and continue
                if (not success or not step_verification['success']) and step.get('critical', False):
                    console.print("[bold red]Critical step failed or didn't achieve its purpose. Stopping execution.[/bold red]")
                    
                    if Confirm.ask("Would you like me to try to fix this and continue?", default=True):
                        fixed = self.fix_failed_step(step, command, output, step_verification)
                        if not fixed:
                            early_termination = True
                            break
                        else:
                            # If we fixed it, update the step result
                            successful_steps.append(step_result)
                            failed_steps.pop()  # Remove from failed steps
                    else:
                        early_termination = True
                        break
        
        console.print("\n[bold green]Plan execution completed.[/bold green]")
        
        # Generate and display summary
        self.generate_execution_summary(task, high_level_plan, executed_steps, successful_steps, failed_steps, early_termination)
    
    def generate_execution_summary(self, task: str, plan: Dict, executed_steps: List[Dict], successful_steps: List[Dict], failed_steps: List[Dict], early_termination: bool = False):
        """
        Generate a summary of the execution results.
        
        Args:
            task: The original task
            plan: The high-level plan that was executed
            executed_steps: List of all steps that were executed
            successful_steps: List of steps that were successful
            failed_steps: List of steps that failed
            early_termination: Whether execution was terminated early
        """
        console.print("\n[bold blue]━━━━━━━━━━━━━━━━ EXECUTION SUMMARY ━━━━━━━━━━━━━━━━[/bold blue]")
        
        # Calculate success metrics
        total_steps = len(plan['steps'])
        executed_count = len(executed_steps)
        successful_count = len(successful_steps)
        
        execution_percentage = int((executed_count / total_steps) * 100) if total_steps > 0 else 0
        success_percentage = int((successful_count / executed_count) * 100) if executed_count > 0 else 0
        
        # Determine overall success
        if early_termination:
            overall_status = "[bold yellow]PARTIAL SUCCESS[/bold yellow]"
        elif failed_steps:
            if successful_count > len(failed_steps):
                overall_status = "[bold yellow]PARTIAL SUCCESS[/bold yellow]"
            else:
                overall_status = "[bold red]FAILED[/bold red]"
        else:
            overall_status = "[bold green]SUCCESS[/bold green]"
        
        # Print summary header
        console.print(f"Task: [bold]{task}[/bold]")
        console.print(f"Status: {overall_status}")
        console.print(f"Steps Executed: {executed_count}/{total_steps} ({execution_percentage}%)")
        console.print(f"Steps Successful: {successful_count}/{executed_count} ({success_percentage}%)")
        
        # Use AI to generate a natural language summary
        summary = self.generate_natural_language_summary(task, plan, executed_steps, successful_steps, failed_steps, early_termination)
        
        console.print("\n[bold blue]Summary:[/bold blue]")
        console.print(f"{summary}")
        
        # Print key findings or outputs if available
        key_outputs = self.extract_key_outputs(executed_steps)
        if key_outputs:
            console.print("\n[bold blue]Key Findings:[/bold blue]")
            for output in key_outputs:
                console.print(f"• {output}")
        
        console.print("\n[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold blue]")
        
        # Log the summary
        self.logger.log_system_message(f"Task execution summary: {overall_status}, {successful_count}/{executed_count} steps successful")
    
    def generate_natural_language_summary(self, task: str, plan: Dict, executed_steps: List[Dict], successful_steps: List[Dict], failed_steps: List[Dict], early_termination: bool) -> str:
        """
        Generate a natural language summary of the execution.
        
        Args:
            task: The original task
            plan: The high-level plan that was executed
            executed_steps: List of all steps that were executed
            successful_steps: List of steps that were successful
            failed_steps: List of steps that failed
            early_termination: Whether execution was terminated early
            
        Returns:
            str: A natural language summary of what was accomplished
        """
        system_message = """You are an AI assistant that summarizes the execution of a task.
        Given information about a task, its execution steps, and results, provide a concise summary of:
        1. What was accomplished
        2. What challenges or failures were encountered, if any
        3. The overall outcome of the task
        
        Keep your summary brief (3-5 sentences) but informative, focusing on the most important aspects.
        """
        
        # Prepare step information
        analysis_steps = [s for s in executed_steps if s.get('phase') == 'analysis']
        planning_steps = [s for s in executed_steps if s.get('phase') == 'planning']
        execution_steps = [s for s in executed_steps if s.get('phase') == 'execution']
        
        step_summaries = []
        for step in executed_steps:
            step_status = "✅ Successful" if step['success'] else "❌ Failed"
            step_summaries.append(f"Step {step['step_number']}: {step['description']} - {step_status}")
        
        steps_info = "\n".join(step_summaries)
        
        # Collect important outputs
        important_outputs = []
        for step in executed_steps:
            # Only include outputs that have meaningful content (not too long, not empty)
            if len(step['output'].strip()) > 0 and len(step['output']) < 500:
                important_outputs.append(f"Step {step['step_number']} output: {step['output'].strip()}")
        
        outputs_info = "\n".join(important_outputs[:5])  # Limit to 5 most important outputs
        
        user_content = f"""Task: {task}

Number of steps in plan: {len(plan['steps'])}
Number of steps executed: {len(executed_steps)}
Number of successful steps: {len(successful_steps)}
Number of failed steps: {len(failed_steps)}
Early termination: {'Yes' if early_termination else 'No'}

STEPS INFORMATION:
{steps_info}

IMPORTANT OUTPUTS:
{outputs_info}

Please provide a concise 3-5 sentence summary of what was accomplished, any challenges encountered, and the overall outcome."""
        
        try:
            # Generate the summary
            response = openai.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            self.log_debug(f"Error generating summary: {str(e)}")
            
            # Provide a basic summary if the API call fails
            executed_percent = int((len(executed_steps) / len(plan['steps'])) * 100) if plan['steps'] else 0
            success_percent = int((len(successful_steps) / len(executed_steps)) * 100) if executed_steps else 0
            
            if early_termination:
                return f"The task was partially completed with {executed_percent}% of steps executed and {success_percent}% of those steps successful. The execution was terminated early due to a critical failure. Some valuable information was gathered but the task could not be fully completed."
            elif len(failed_steps) == 0:
                return f"The task was completed successfully with all {len(executed_steps)} steps executed as planned. All steps achieved their intended outcomes."
            else:
                return f"The task was partially completed with {executed_percent}% of steps executed and {success_percent}% of those steps successful. Some steps failed to achieve their intended outcomes, but valuable information was still gathered."
    
    def extract_key_outputs(self, executed_steps: List[Dict]) -> List[str]:
        """
        Extract key findings or outputs from the executed steps.
        
        Args:
            executed_steps: List of executed steps with their outputs
            
        Returns:
            List[str]: Important outputs or findings
        """
        # First, collect all outputs that might contain useful information
        candidate_outputs = []
        
        for step in executed_steps:
            if step['success'] and step['output'].strip():
                # Prioritize execution phase outputs
                priority = 1 if step['phase'] == 'execution' else 2
                candidate_outputs.append({
                    "step": step['step_number'],
                    "output": step['output'].strip(),
                    "description": step['description'],
                    "priority": priority
                })
        
        # Simple heuristic: Outputs that are not too short, not too long
        filtered_outputs = []
        for output in candidate_outputs:
            lines = output['output'].split('\n')
            if 0 < len(lines) < 20:  # Not too many lines
                filtered_outputs.append(output)
        
        # Sort by priority (execution steps first) and limit to 5
        filtered_outputs.sort(key=lambda x: x['priority'])
        
        # Format the outputs for display
        key_outputs = []
        for output in filtered_outputs[:5]:
            # Try to summarize the output if it's more than 3 lines
            lines = output['output'].split('\n')
            if len(lines) > 3:
                summary = f"Step {output['step']} ({output['description']}): Found {len(lines)} items"
                # Add a sample of items if appropriate
                if len(lines) < 10:
                    sample = ", ".join(line.strip() for line in lines[:3])
                    summary += f" including {sample}..."
                key_outputs.append(summary)
            else:
                # Short outputs can be included directly
                key_outputs.append(f"Step {output['step']} ({output['description']}): {output['output']}")
        
        return key_outputs
    
    def handle_content_creation_task(self, task: str):
        """
        Handle content creation tasks like writing reports, articles, etc.
        
        Args:
            task: The content creation task description
        """
        self.logger.log_system_message(f"Creating content for task: {task}")
        console.print("[blue]Planning content structure...[/blue]")
        
        # Generate content outline first
        outline = self.generate_content_outline(task)
        
        # Display the outline to the user
        console.print("\n[bold blue]Content Outline:[/bold blue]")
        console.print(Markdown(outline))
        
        # Ask for confirmation before proceeding with full content generation
        if self.mode == "manual" and not Confirm.ask("Generate content based on this outline?", default=True):
            console.print("[blue]Content generation cancelled.[/blue]")
            return
        
        # Generate the full content
        console.print("\n[blue]Generating content... (this may take a few moments)[/blue]")
        content = self.generate_full_content(task, outline)
        
        # Display the content to the user
        console.print("\n[bold blue]Generated Content:[/bold blue]")
        console.print(Markdown(content))
        
        # Log the generated content
        self.logger.log_system_message("Content generated successfully")
        
        # Ask if user wants to save the content to a file
        if Confirm.ask("Would you like to save this content to a file?", default=True):
            filename = self.get_filename_for_content(task)
            self.save_content_to_file(content, filename)
    
    def generate_content_outline(self, task: str) -> str:
        """
        Generate an outline for content creation tasks.
        
        Args:
            task: Content creation task description
            
        Returns:
            str: Content outline in markdown format
        """
        system_message = """You are an AI assistant that creates detailed outlines for content.
        Create a comprehensive outline for the requested content.
        Format your outline in Markdown with clear sections and subsections.
        Be thorough but concise in your outline structure.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Create a detailed outline for the following content task: {task}"}
        ]
        
        try:
            # Use the animation helper for the API call
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.3
            )
            
            outline = response.choices[0].message.content
            self.logger.log_system_message("Content outline generated")
            return outline
            
        except Exception as e:
            error_msg = f"Error generating content outline: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(error_msg)
            return "# Error in Outline Generation\n\nFailed to create outline."
    
    def generate_full_content(self, task: str, outline: str) -> str:
        """
        Generate full content based on task and outline.
        
        Args:
            task: Content creation task description
            outline: Content outline
            
        Returns:
            str: Generated content in markdown format
        """
        system_message = """You are an AI assistant that creates detailed, well-researched content.
        Create comprehensive content based on the provided task and outline.
        Make sure to follow the outline structure but expand with detailed information.
        Format your response using Markdown for better readability.
        Include proper headings, subheadings, bullet points, and formatting.
        Aim to be informative, accurate, and well-structured.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Create detailed content for the following task: {task}\n\nUse this outline as the structure:\n\n{outline}"}
        ]
        
        try:
            # Use the animation helper for the API call
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            self.logger.log_system_message("Full content generated")
            return content
            
        except Exception as e:
            error_msg = f"Error generating full content: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(error_msg)
            return "# Error in Content Generation\n\nFailed to create content."
    
    def get_filename_for_content(self, task: str) -> str:
        """
        Generate a suitable filename for the content.
        
        Args:
            task: The content task
            
        Returns:
            str: A suitable filename
        """
        # Sanitize task to create filename
        sanitized = re.sub(r'[^\w\s-]', '', task.lower())
        sanitized = re.sub(r'[\s-]+', '_', sanitized)
        base_filename = sanitized[:50]  # Limit length
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_filename}_{timestamp}.md"
        
        # Allow user to modify the filename
        user_filename = Prompt.ask("[bold]Enter filename to save content[/bold]", default=filename)
        
        # Ensure .md extension
        if not user_filename.endswith('.md'):
            user_filename += '.md'
            
        return user_filename
    
    def save_content_to_file(self, content: str, filename: str):
        """
        Save generated content to a file.
        
        Args:
            content: The content to save
            filename: Filename to save to
        """
        try:
            # Create full path
            file_path = os.path.join(os.getcwd(), filename)
            
            # Write content to file
            with open(file_path, 'w') as f:
                f.write(content)
                
            console.print(f"[bold green]Content saved to [/bold green][bold cyan]{file_path}[/bold cyan]")
            self.logger.log_system_message(f"Content saved to file: {file_path}")
            
            # Execute a command to show the file was created
            self.execute_command(f"ls -la {shlex.quote(file_path)}")
            
        except Exception as e:
            error_msg = f"Error saving content to file: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(error_msg)
    
    def generate_plan(self, task: str) -> Dict:
        """
        Generate a step-by-step plan for a complex task.
        
        Args:
            task: Complex task description
            
        Returns:
            Dict: Plan in JSON format with an array of steps
        """
        # Get step-by-step plan from OpenAI
        system_message = """You are an AI assistant that breaks down complex tasks into a series of shell commands for macOS.
        Your approach should mirror human-like reasoning by ALWAYS following these phases:

        PHASE 1: ANALYSIS & DISCOVERY
        - List and examine what exists in the environment
        - Categorize what you find (file types, structures, patterns)
        - Understand the current state before making changes
        
        PHASE 2: PLANNING
        - Based on your analysis, develop a thoughtful approach
        - Consider multiple ways to accomplish the goal
        - Choose the most appropriate method based on context

        PHASE 3: EXECUTION
        - Only after thorough analysis, create commands to accomplish the goal
        - Include verification steps to confirm actions worked as expected
        
        Create a step-by-step plan with specific shell commands to accomplish the user's goal.
        Format your response as JSON with an array of steps, each with:
        - description: Brief description of what this step does
        - command: The exact shell command to run
        - critical: Boolean indicating if this step is critical (failure should stop execution)
        - phase: String indicating which phase this step belongs to ("analysis", "planning", or "execution")
        
        IMPORTANT GUIDELINES FOR COMMAND EXECUTION:
        
        1. DIRECTORY PERSISTENCE: Each command is executed in a separate subprocess, so directory changes with 'cd' DO NOT persist between steps.
           When working with directories, choose ONE of these approaches:
           - Use absolute paths in commands
           - Combine 'cd' and the actual command in the same step with '&&' (e.g., "cd /path && command")
           - Use relative paths from the initial directory

        2. When writing files to a new directory, either:
           - Use the full path: "echo content > /path/to/dir/file.txt"
           - Combine commands: "cd /path/to/dir && echo content > file.txt"
           
        3. Avoid unnecessary 'cd' commands when you can specify the full path directly
        
        4. For multi-part operations in the same directory, consider combining them with '&&' in a single step
        
        EXAMPLE OF A WELL-STRUCTURED PLAN:
        ```
        {
          "steps": [
            {
              "description": "Analyze what files and directories exist in the current location",
              "command": "find . -type f -o -type d | sort",
              "critical": true,
              "phase": "analysis"
            },
            {
              "description": "Examine file types to understand what we're working with",
              "command": "find . -type f -exec file {} \\;",
              "critical": true,
              "phase": "analysis"
            },
            {
              "description": "Check for large files that might need special handling",
              "command": "find . -type f -size +10M -exec ls -lh {} \\;",
              "critical": false,
              "phase": "analysis"
            },
            {
              "description": "Based on analysis, create a directory structure for organization",
              "command": "mkdir -p media/images documents/text data/csv",
              "critical": true,
              "phase": "planning"
            },
            {
              "description": "Move image files to the images directory",
              "command": "find . -maxdepth 1 -type f -name '*.jpg' -o -name '*.png' -exec mv {} ./media/images/ \\;",
              "critical": true,
              "phase": "execution"
            },
            {
              "description": "Verify files were moved successfully",
              "command": "ls -la ./media/images/",
              "critical": false,
              "phase": "execution"
            }
          ]
        }
        ```
        """
        
        # Get directory context
        dir_context = self.get_directory_context()
        
        user_content = f"""Task: {task}

Directory Information:
{dir_context}

IMPORTANT NOTE: Remember that each command runs separately, so directory changes with 'cd' won't persist across steps. 
Use absolute paths, combined commands with &&, or ensure your paths account for this limitation.

REMEMBER: Follow the human-like reasoning process by first analyzing what exists, then planning, and finally executing the plan."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        try:
            # Show thinking animation while creating the plan
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            return plan
            
        except Exception as e:
            console.print(f"[bold red]Error generating plan: {str(e)}[/bold red]")
            if self.debug:
                import traceback
                console.print(traceback.format_exc())
            return None
    
    def verify_and_refine_plan(self, task: str, initial_plan: Dict) -> Dict:
        """
        Verify and refine a plan through multiple iterations.
        
        Args:
            task: Original task description
            initial_plan: Initial plan generated
            
        Returns:
            Dict: Final verified and refined plan
        """
        current_plan = initial_plan
        iterations = 0
        
        console.print("[blue]Verifying and refining the plan...[/blue]")
        
        while iterations < self.max_plan_iterations:
            # Create a progress message
            if iterations > 0:
                console.print(f"[dim]Plan refinement iteration {iterations+1}/{self.max_plan_iterations}[/dim]")
            
            # Verify the current plan
            verification_result = self.verify_plan(task, current_plan)
            
            # If the plan is verified as good, break the loop
            if verification_result.get('is_good', False):
                console.print("[green]Plan verification completed: Plan is sound.[/green]")
                break
                
            # If refinements are suggested, update the plan
            if 'refined_plan' in verification_result and verification_result['refined_plan']:
                # Check if the refined plan is different from the current one
                if verification_result['refined_plan'] != current_plan:
                    console.print("[blue]Refining plan based on verification feedback...[/blue]")
                    current_plan = verification_result['refined_plan']
                else:
                    # No changes in the plan, break the loop
                    console.print("[green]Plan verification completed: No further refinements needed.[/green]")
                    break
            else:
                # No refinements suggested, break the loop
                console.print("[yellow]Plan verification completed: No refinements suggested.[/yellow]")
                break
                
            iterations += 1
            
        if iterations >= self.max_plan_iterations:
            console.print("[yellow]Reached maximum plan refinement iterations.[/yellow]")
            
        return current_plan
    
    def verify_plan(self, task: str, plan: Dict) -> Dict:
        """
        Verify a plan using OpenAI API.
        
        Args:
            task: Original task description
            plan: Plan to verify
            
        Returns:
            Dict: Verification result with is_good flag and possibly a refined_plan
        """
        # Get directory context
        dir_context = self.get_directory_context()
        
        # Get conversation history
        conv_history = self.logger.get_conversation_history()
        
        # Create a system message for plan verification
        system_message = """You are an AI assistant that verifies the soundness of shell command execution plans.
        Analyze the given plan and task to determine if:
        1. The plan accomplishes the task completely
        2. The commands are correct and efficient
        3. The steps are in the right order
        4. There are no missing steps
        5. There are no unnecessary steps
        
        IMPORTANT: The plan must follow a human-like reasoning process with these phases:
        
        PHASE 1: ANALYSIS & DISCOVERY
        - At least 2-3 steps should be dedicated to analyzing what exists before taking action
        - The plan should include commands to list and examine files/directories
        - There should be steps to categorize or understand the content (file types, sizes, etc.)
        
        PHASE 2: PLANNING
        - Based on the analysis, the plan should include steps for organizing the approach
        - This may include creating directory structures or determining what needs to be done
        
        PHASE 3: EXECUTION
        - Only after thorough analysis and planning should the actual task be executed
        - Verification steps should be included to confirm actions worked as expected
        
        CRITICAL ISSUES TO CHECK:
        
        - Directory persistence: When the plan includes a 'cd' command, subsequent commands will NOT inherit the directory change when executed separately.
          This is because each command runs in its own subprocess. Fix this by either:
          a) Combining steps with directory changes using && (e.g., "cd dir && command")
          b) Using absolute paths in subsequent commands
          c) Prepending the target directory to all file operations (e.g., "echo content > dir/file" instead of "cd dir" then "echo content > file")
        
        - Command dependencies: Ensure that commands which depend on previous commands' outputs are properly sequenced
        
        - Error handling: Consider adding checks to verify critical steps succeeded before proceeding
        
        - Permission issues: Check if commands might require permissions (e.g., sudo) or create files in protected directories
        
        - File overwriting: Be cautious about commands that might overwrite existing files without confirmation
        
        If improvements are needed, provide a refined plan in the same format.
        Format your response as JSON with:
        - is_good: Boolean indicating if the plan is sound (true) or needs refinement (false)
        - feedback: Brief explanation of your assessment, particularly highlighting issues like directory persistence or missing analysis steps
        - refined_plan: The improved plan if is_good is false, otherwise null
        """
        
        # Convert plan to a formatted string for readability
        plan_text = json.dumps(plan, indent=2)
        
        # Create the user message
        user_content = f"""Task: {task}

Proposed Plan:
{plan_text}

Directory Information:
{dir_context}

Recent Conversation History:
{conv_history}

Please verify if this plan is sound and complete for accomplishing the task. 
The plan MUST follow a human-like reasoning process with proper analysis steps before taking action.
Pay special attention to directory persistence issues - if any step changes directory with 'cd', subsequent commands need to account for this by either using absolute paths or combining commands with &&."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        try:
            # Call the API with animation
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Display feedback
            if 'feedback' in result:
                console.print(f"[dim]Verification feedback: {result['feedback']}[/dim]")
                self.logger.log_system_message(f"Plan verification feedback: {result['feedback']}")
                
            return result
            
        except Exception as e:
            error_msg = f"Error verifying plan: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(f"Error: {error_msg}")
            # Return a default result indicating the plan is good to avoid blocking execution
            return {"is_good": True, "feedback": "Verification failed, proceeding with original plan."}
    
    def get_directory_context(self) -> str:
        """
        Get context information about the current directory environment.
        
        Returns:
            str: Context information about current directory and environment
        """
        try:
            # Get current directory path
            current_dir = os.getcwd()
            
            # Get directory contents (files and folders)
            dir_contents = os.listdir(current_dir)
            # Limit the list to avoid excessive information
            if len(dir_contents) > 10:
                dir_contents = dir_contents[:10] + ["..."]
            
            # Get parent directory
            parent_dir = os.path.dirname(current_dir)
            
            # Construct the context information
            context = f"Current directory: {current_dir}\n"
            context += f"Parent directory: {parent_dir}\n"
            context += f"Directory contents: {', '.join(dir_contents)}\n"
            
            return context
        except Exception as e:
            self.log_debug(f"Error getting directory context: {str(e)}")
            return "Unable to determine directory context."
            
    def call_api_with_animation(self, api_function, **kwargs):
        """
        Call an API with a loading animation using Rich.
        
        Args:
            api_function: Function reference to the API call
            **kwargs: Keyword arguments to pass to the API function
            
        Returns:
            The API response
        """
        spinner_text = "Thinking..."
        spinner_style = "bold blue"
        
        # Show a spinner animation while waiting for the API response
        with console.status(f"[{spinner_style}]{spinner_text}[/{spinner_style}]", spinner="dots") as status:
            try:
                # Call the API function with the provided arguments
                response = api_function(**kwargs)
                return response
            except Exception as e:
                # Log any errors but don't display to user yet (caller will handle that)
                self.log_debug(f"API call error: {str(e)}")
                raise
    
    def is_shell_command(self, user_input: str) -> bool:
        """
        Determine if user input is likely a shell command that can be executed directly.
        This method is kept for backward compatibility but now uses the API-based approach.
        
        Args:
            user_input: User's input
            
        Returns:
            bool: True if the input appears to be a shell command, False otherwise
        """
        # Use the API-based classification instead of rule-based patterns
        intent = self.classify_user_intent(user_input)
        
        # Log the classification result
        self.log_debug(f"Shell command check via API: {intent['primary_intent']}")
        
        # Return true if the API classified this as a direct command
        return intent['primary_intent'] == 'direct_command'
    
    def _is_in_quotes(self, text: str, substring: str) -> bool:
        """
        Check if a substring appears only inside quotes in the text.
        
        Args:
            text: The full text string
            substring: The substring to check
            
        Returns:
            bool: True if the substring only appears inside quotes, False otherwise
        """
        in_single_quotes = False
        in_double_quotes = False
        
        for i in range(len(text)):
            # Toggle quote state
            if text[i] == "'" and not in_double_quotes:
                in_single_quotes = not in_single_quotes
            elif text[i] == '"' and not in_single_quotes:
                in_double_quotes = not in_double_quotes
                
            # Check if we're at the start of our substring
            if i <= len(text) - len(substring) and text[i:i+len(substring)] == substring:
                # If we're not in quotes, the substring is not exclusively in quotes
                if not in_single_quotes and not in_double_quotes:
                    return False
        
        # If we've found instances but they were all in quotes
        return True
    
    def refine_user_query(self, user_input: str) -> str:
        """
        Refine a user query to make it more precise for command generation.
        
        Args:
            user_input: Original user query
            
        Returns:
            str: Refined user query
        """
        if not self.refine_queries:
            return user_input
            
        self.log_debug(f"Refining user query: {user_input}")
        
        # Get directory context
        dir_context = self.get_directory_context()
        
        # Get conversation history for additional context
        conversation_history = self.logger.get_conversation_history()
        self.logger.log_system_message("Using conversation history for query refinement")
        
        # Create a system message for query refinement
        system_message = """You are an AI assistant that refines user queries into clear, specific instructions.
        Your task is to clarify ambiguous requests and add necessary context.
        Do NOT generate commands directly - just improve the query for clarity.
        Keep the refined query brief and focused.
        Consider the conversation history to understand the user's current context and needs.
        """
        
        # Create the messages for the API call
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Refine this query for a command-line assistant: '{user_input}'\n\nDirectory Information:\n{dir_context}\n\nRecent Conversation History:\n{conversation_history}"}
        ]
        
        try:
            # Call OpenAI API with the request
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.3
            )
            
            # Extract the refined query
            refined_query = response.choices[0].message.content.strip()
            self.log_debug(f"Refined query: {refined_query}")
            
            # If the refinement significantly changes the query, let the user know
            if len(refined_query) > len(user_input) * 1.5 or len(refined_query) < len(user_input) * 0.5:
                console.print(f"[dim]Refined query: {refined_query}[/dim]")
                
            return refined_query
            
        except Exception as e:
            # Log error but continue with original query
            self.log_debug(f"Error refining query: {str(e)}")
            return user_input
            
    def classify_user_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Unified function to classify user input intent using the API,
        replacing multiple rule-based detection methods.
        
        Args:
            user_input: User's input text
            
        Returns:
            Dict[str, Any]: Classification results with intent categories and confidence scores
        """
        self.log_debug(f"Classifying user intent: {user_input}")
        
        # Get context information
        dir_context = self.get_directory_context()
        conversation_history = self.logger.get_conversation_history()
        
        # Create a system message for intent classification
        system_message = """You are an AI assistant that classifies user input intent for a terminal assistant.
        Your task is to analyze user input and determine the type of request.
        
        Respond with a JSON object containing the following fields:
        - "primary_intent": One of the following categories:
          * "general_information" - General information question that doesn't require command execution
          * "content_creation" - Writing, reporting, documentation, analysis
          * "complex_task" - Task requiring multiple commands/steps
          * "context_request" - User asking about previous actions/commands
          * "direct_command" - A specific shell command
          * "simple_task" - Simple task requiring a single command
          
        - "confidence": Confidence score (0-1) for the primary intent
        - "dangerous_command": Boolean indicating if this might be a dangerous command (if applicable)
        - "risk_level": Number from 0-10 indicating potential risk level of the command (if applicable)
        - "risk_reasons": List of reasons why the command is risky (if applicable)
        - "requires_agent_mode": Boolean indicating if this task should use agent mode
        - "analysis": Brief explanation of why this classification was chosen
        
        When assessing command danger, consider:
        - Data deletion or modification risks
        - System-wide changes
        - Privilege escalation
        - Resource exhaustion
        - Security implications
        - Network and external access
        - Irreversible actions
        
        Base your classification on the specific request, directory context, and conversation history.
        """
        
        # Create the messages for the API call
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Classify this user input: '{user_input}'\n\nDirectory Information:\n{dir_context}\n\nRecent Conversation History:\n{conversation_history}"}
        ]
        
        try:
            # Call OpenAI API with the request
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            # Extract and parse the classification result
            classification = json.loads(response.choices[0].message.content)
            self.log_debug(f"Intent classification result: {classification}")
            
            # Log the classification decision
            self.logger.log_system_message(f"Intent classification: {classification['primary_intent']} (confidence: {classification['confidence']})")
            if 'analysis' in classification:
                self.logger.log_system_message(f"Analysis: {classification['analysis']}")
            
            # Log risk assessment if available
            if 'risk_level' in classification:
                self.logger.log_system_message(f"Risk assessment: Level {classification['risk_level']}/10")
                if 'risk_reasons' in classification and classification['risk_reasons']:
                    self.logger.log_system_message(f"Risk reasons: {', '.join(classification['risk_reasons'])}")
            
            return classification
            
        except Exception as e:
            # Log error and return a default classification
            error_msg = f"Error in intent classification: {str(e)}"
            self.log_debug(error_msg)
            self.logger.log_system_message(error_msg)
            
            # Return a conservative default
            return {
                "primary_intent": "simple_task",  # Default to simple task
                "confidence": 0.5,
                "dangerous_command": False,
                "requires_agent_mode": False,
                "analysis": "Classification failed, defaulting to simple task"
            }
    
    def extract_key_outputs(self, executed_steps: List[Dict]) -> List[str]:
        """
        Extract key findings or outputs from the executed steps.
        
        Args:
            executed_steps: List of executed steps with their outputs
            
        Returns:
            List[str]: Important outputs or findings
        """
        # First, collect all outputs that might contain useful information
        candidate_outputs = []
        
        for step in executed_steps:
            if step['success'] and step['output'].strip():
                # Prioritize execution phase outputs
                priority = 1 if step['phase'] == 'execution' else 2
                candidate_outputs.append({
                    "step": step['step_number'],
                    "output": step['output'].strip(),
                    "description": step['description'],
                    "priority": priority
                })
        
        # Simple heuristic: Outputs that are not too short, not too long
        filtered_outputs = []
        for output in candidate_outputs:
            lines = output['output'].split('\n')
            if 0 < len(lines) < 20:  # Not too many lines
                filtered_outputs.append(output)
        
        # Sort by priority (execution steps first) and limit to 5
        filtered_outputs.sort(key=lambda x: x['priority'])
        
        # Format the outputs for display
        key_outputs = []
        for output in filtered_outputs[:5]:
            # Try to summarize the output if it's more than 3 lines
            lines = output['output'].split('\n')
            if len(lines) > 3:
                summary = f"Step {output['step']} ({output['description']}): Found {len(lines)} items"
                # Add a sample of items if appropriate
                if len(lines) < 10:
                    sample = ", ".join(line.strip() for line in lines[:3])
                    summary += f" including {sample}..."
                key_outputs.append(summary)
            else:
                # Short outputs can be included directly
                key_outputs.append(f"Step {output['step']} ({output['description']}): {output['output']}")
        
        return key_outputs
    
    def handle_content_creation_task(self, task: str):
        """
        Handle content creation tasks like writing reports, articles, etc.
        
        Args:
            task: The content creation task description
        """
        self.logger.log_system_message(f"Creating content for task: {task}")
        console.print("[blue]Planning content structure...[/blue]")
        
        # Generate content outline first
        outline = self.generate_content_outline(task)
        
        # Display the outline to the user
        console.print("\n[bold blue]Content Outline:[/bold blue]")
        console.print(Markdown(outline))
        
        # Ask for confirmation before proceeding with full content generation
        if self.mode == "manual" and not Confirm.ask("Generate content based on this outline?", default=True):
            console.print("[blue]Content generation cancelled.[/blue]")
            return
        
        # Generate the full content
        console.print("\n[blue]Generating content... (this may take a few moments)[/blue]")
        content = self.generate_full_content(task, outline)
        
        # Display the content to the user
        console.print("\n[bold blue]Generated Content:[/bold blue]")
        console.print(Markdown(content))
        
        # Log the generated content
        self.logger.log_system_message("Content generated successfully")
        
        # Ask if user wants to save the content to a file
        if Confirm.ask("Would you like to save this content to a file?", default=True):
            filename = self.get_filename_for_content(task)
            self.save_content_to_file(content, filename)
    
    def generate_content_outline(self, task: str) -> str:
        """
        Generate an outline for content creation tasks.
        
        Args:
            task: Content creation task description
            
        Returns:
            str: Content outline in markdown format
        """
        system_message = """You are an AI assistant that creates detailed outlines for content.
        Create a comprehensive outline for the requested content.
        Format your outline in Markdown with clear sections and subsections.
        Be thorough but concise in your outline structure.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Create a detailed outline for the following content task: {task}"}
        ]
        
        try:
            # Use the animation helper for the API call
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.3
            )
            
            outline = response.choices[0].message.content
            self.logger.log_system_message("Content outline generated")
            return outline
            
        except Exception as e:
            error_msg = f"Error generating content outline: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(error_msg)
            return "# Error in Outline Generation\n\nFailed to create outline."
    
    def generate_full_content(self, task: str, outline: str) -> str:
        """
        Generate full content based on task and outline.
        
        Args:
            task: Content creation task description
            outline: Content outline
            
        Returns:
            str: Generated content in markdown format
        """
        system_message = """You are an AI assistant that creates detailed, well-researched content.
        Create comprehensive content based on the provided task and outline.
        Make sure to follow the outline structure but expand with detailed information.
        Format your response using Markdown for better readability.
        Include proper headings, subheadings, bullet points, and formatting.
        Aim to be informative, accurate, and well-structured.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Create detailed content for the following task: {task}\n\nUse this outline as the structure:\n\n{outline}"}
        ]
        
        try:
            # Use the animation helper for the API call
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            self.logger.log_system_message("Full content generated")
            return content
            
        except Exception as e:
            error_msg = f"Error generating full content: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(error_msg)
            return "# Error in Content Generation\n\nFailed to create content."
    
    def get_filename_for_content(self, task: str) -> str:
        """
        Generate a suitable filename for the content.
        
        Args:
            task: The content task
            
        Returns:
            str: A suitable filename
        """
        # Sanitize task to create filename
        sanitized = re.sub(r'[^\w\s-]', '', task.lower())
        sanitized = re.sub(r'[\s-]+', '_', sanitized)
        base_filename = sanitized[:50]  # Limit length
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_filename}_{timestamp}.md"
        
        # Allow user to modify the filename
        user_filename = Prompt.ask("[bold]Enter filename to save content[/bold]", default=filename)
        
        # Ensure .md extension
        if not user_filename.endswith('.md'):
            user_filename += '.md'
            
        return user_filename
    
    def save_content_to_file(self, content: str, filename: str):
        """
        Save generated content to a file.
        
        Args:
            content: The content to save
            filename: Filename to save to
        """
        try:
            # Create full path
            file_path = os.path.join(os.getcwd(), filename)
            
            # Write content to file
            with open(file_path, 'w') as f:
                f.write(content)
                
            console.print(f"[bold green]Content saved to [/bold green][bold cyan]{file_path}[/bold cyan]")
            self.logger.log_system_message(f"Content saved to file: {file_path}")
            
            # Execute a command to show the file was created
            self.execute_command(f"ls -la {shlex.quote(file_path)}")
            
        except Exception as e:
            error_msg = f"Error saving content to file: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(error_msg)
    
    def generate_plan(self, task: str) -> Dict:
        """
        Generate a step-by-step plan for a complex task.
        
        Args:
            task: Complex task description
            
        Returns:
            Dict: Plan in JSON format with an array of steps
        """
        # Get step-by-step plan from OpenAI
        system_message = """You are an AI assistant that breaks down complex tasks into a series of shell commands for macOS.
        Your approach should mirror human-like reasoning by ALWAYS following these phases:

        PHASE 1: ANALYSIS & DISCOVERY
        - List and examine what exists in the environment
        - Categorize what you find (file types, structures, patterns)
        - Understand the current state before making changes
        
        PHASE 2: PLANNING
        - Based on your analysis, develop a thoughtful approach
        - Consider multiple ways to accomplish the goal
        - Choose the most appropriate method based on context

        PHASE 3: EXECUTION
        - Only after thorough analysis, create commands to accomplish the goal
        - Include verification steps to confirm actions worked as expected
        
        Create a step-by-step plan with specific shell commands to accomplish the user's goal.
        Format your response as JSON with an array of steps, each with:
        - description: Brief description of what this step does
        - command: The exact shell command to run
        - critical: Boolean indicating if this step is critical (failure should stop execution)
        - phase: String indicating which phase this step belongs to ("analysis", "planning", or "execution")
        
        IMPORTANT GUIDELINES FOR COMMAND EXECUTION:
        
        1. DIRECTORY PERSISTENCE: Each command is executed in a separate subprocess, so directory changes with 'cd' DO NOT persist between steps.
           When working with directories, choose ONE of these approaches:
           - Use absolute paths in commands
           - Combine 'cd' and the actual command in the same step with '&&' (e.g., "cd /path && command")
           - Use relative paths from the initial directory

        2. When writing files to a new directory, either:
           - Use the full path: "echo content > /path/to/dir/file.txt"
           - Combine commands: "cd /path/to/dir && echo content > file.txt"
           
        3. Avoid unnecessary 'cd' commands when you can specify the full path directly
        
        4. For multi-part operations in the same directory, consider combining them with '&&' in a single step
        
        EXAMPLE OF A WELL-STRUCTURED PLAN:
        ```
        {
          "steps": [
            {
              "description": "Analyze what files and directories exist in the current location",
              "command": "find . -type f -o -type d | sort",
              "critical": true,
              "phase": "analysis"
            },
            {
              "description": "Examine file types to understand what we're working with",
              "command": "find . -type f -exec file {} \\;",
              "critical": true,
              "phase": "analysis"
            },
            {
              "description": "Check for large files that might need special handling",
              "command": "find . -type f -size +10M -exec ls -lh {} \\;",
              "critical": false,
              "phase": "analysis"
            },
            {
              "description": "Based on analysis, create a directory structure for organization",
              "command": "mkdir -p media/images documents/text data/csv",
              "critical": true,
              "phase": "planning"
            },
            {
              "description": "Move image files to the images directory",
              "command": "find . -maxdepth 1 -type f -name '*.jpg' -o -name '*.png' -exec mv {} ./media/images/ \\;",
              "critical": true,
              "phase": "execution"
            },
            {
              "description": "Verify files were moved successfully",
              "command": "ls -la ./media/images/",
              "critical": false,
              "phase": "execution"
            }
          ]
        }
        ```
        """
        
        # Get directory context
        dir_context = self.get_directory_context()
        
        user_content = f"""Task: {task}

Directory Information:
{dir_context}

IMPORTANT NOTE: Remember that each command runs separately, so directory changes with 'cd' won't persist across steps. 
Use absolute paths, combined commands with &&, or ensure your paths account for this limitation.

REMEMBER: Follow the human-like reasoning process by first analyzing what exists, then planning, and finally executing the plan."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        try:
            # Show thinking animation while creating the plan
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            return plan
            
        except Exception as e:
            console.print(f"[bold red]Error generating plan: {str(e)}[/bold red]")
            if self.debug:
                import traceback
                console.print(traceback.format_exc())
            return None
    
    def verify_and_refine_plan(self, task: str, initial_plan: Dict) -> Dict:
        """
        Verify and refine a plan through multiple iterations.
        
        Args:
            task: Original task description
            initial_plan: Initial plan generated
            
        Returns:
            Dict: Final verified and refined plan
        """
        current_plan = initial_plan
        iterations = 0
        
        console.print("[blue]Verifying and refining the plan...[/blue]")
        
        while iterations < self.max_plan_iterations:
            # Create a progress message
            if iterations > 0:
                console.print(f"[dim]Plan refinement iteration {iterations+1}/{self.max_plan_iterations}[/dim]")
            
            # Verify the current plan
            verification_result = self.verify_plan(task, current_plan)
            
            # If the plan is verified as good, break the loop
            if verification_result.get('is_good', False):
                console.print("[green]Plan verification completed: Plan is sound.[/green]")
                break
                
            # If refinements are suggested, update the plan
            if 'refined_plan' in verification_result and verification_result['refined_plan']:
                # Check if the refined plan is different from the current one
                if verification_result['refined_plan'] != current_plan:
                    console.print("[blue]Refining plan based on verification feedback...[/blue]")
                    current_plan = verification_result['refined_plan']
                else:
                    # No changes in the plan, break the loop
                    console.print("[green]Plan verification completed: No further refinements needed.[/green]")
                    break
            else:
                # No refinements suggested, break the loop
                console.print("[yellow]Plan verification completed: No refinements suggested.[/yellow]")
                break
                
            iterations += 1
            
        if iterations >= self.max_plan_iterations:
            console.print("[yellow]Reached maximum plan refinement iterations.[/yellow]")
            
        return current_plan
    
    def verify_plan(self, task: str, plan: Dict) -> Dict:
        """
        Verify a plan using OpenAI API.
        
        Args:
            task: Original task description
            plan: Plan to verify
            
        Returns:
            Dict: Verification result with is_good flag and possibly a refined_plan
        """
        # Get directory context
        dir_context = self.get_directory_context()
        
        # Get conversation history
        conv_history = self.logger.get_conversation_history()
        
        # Create a system message for plan verification
        system_message = """You are an AI assistant that verifies the soundness of shell command execution plans.
        Analyze the given plan and task to determine if:
        1. The plan accomplishes the task completely
        2. The commands are correct and efficient
        3. The steps are in the right order
        4. There are no missing steps
        5. There are no unnecessary steps
        
        IMPORTANT: The plan must follow a human-like reasoning process with these phases:
        
        PHASE 1: ANALYSIS & DISCOVERY
        - At least 2-3 steps should be dedicated to analyzing what exists before taking action
        - The plan should include commands to list and examine files/directories
        - There should be steps to categorize or understand the content (file types, sizes, etc.)
        
        PHASE 2: PLANNING
        - Based on the analysis, the plan should include steps for organizing the approach
        - This may include creating directory structures or determining what needs to be done
        
        PHASE 3: EXECUTION
        - Only after thorough analysis and planning should the actual task be executed
        - Verification steps should be included to confirm actions worked as expected
        
        CRITICAL ISSUES TO CHECK:
        
        - Directory persistence: When the plan includes a 'cd' command, subsequent commands will NOT inherit the directory change when executed separately.
          This is because each command runs in its own subprocess. Fix this by either:
          a) Combining steps with directory changes using && (e.g., "cd dir && command")
          b) Using absolute paths in subsequent commands
          c) Prepending the target directory to all file operations (e.g., "echo content > dir/file" instead of "cd dir" then "echo content > file")
        
        - Command dependencies: Ensure that commands which depend on previous commands' outputs are properly sequenced
        
        - Error handling: Consider adding checks to verify critical steps succeeded before proceeding
        
        - Permission issues: Check if commands might require permissions (e.g., sudo) or create files in protected directories
        
        - File overwriting: Be cautious about commands that might overwrite existing files without confirmation
        
        If improvements are needed, provide a refined plan in the same format.
        Format your response as JSON with:
        - is_good: Boolean indicating if the plan is sound (true) or needs refinement (false)
        - feedback: Brief explanation of your assessment, particularly highlighting issues like directory persistence or missing analysis steps
        - refined_plan: The improved plan if is_good is false, otherwise null
        """
        
        # Convert plan to a formatted string for readability
        plan_text = json.dumps(plan, indent=2)
        
        # Create the user message
        user_content = f"""Task: {task}

Proposed Plan:
{plan_text}

Directory Information:
{dir_context}

Recent Conversation History:
{conv_history}

Please verify if this plan is sound and complete for accomplishing the task. 
The plan MUST follow a human-like reasoning process with proper analysis steps before taking action.
Pay special attention to directory persistence issues - if any step changes directory with 'cd', subsequent commands need to account for this by either using absolute paths or combining commands with &&."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        try:
            # Call the API with animation
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Display feedback
            if 'feedback' in result:
                console.print(f"[dim]Verification feedback: {result['feedback']}[/dim]")
                self.logger.log_system_message(f"Plan verification feedback: {result['feedback']}")
                
            return result
            
        except Exception as e:
            error_msg = f"Error verifying plan: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(f"Error: {error_msg}")
            # Return a default result indicating the plan is good to avoid blocking execution
            return {"is_good": True, "feedback": "Verification failed, proceeding with original plan."}
    
    def get_directory_context(self) -> str:
        """
        Get context information about the current directory environment.
        
        Returns:
            str: Context information about current directory and environment
        """
        try:
            # Get current directory path
            current_dir = os.getcwd()
            
            # Get directory contents (files and folders)
            dir_contents = os.listdir(current_dir)
            # Limit the list to avoid excessive information
            if len(dir_contents) > 10:
                dir_contents = dir_contents[:10] + ["..."]
            
            # Get parent directory
            parent_dir = os.path.dirname(current_dir)
            
            # Construct the context information
            context = f"Current directory: {current_dir}\n"
            context += f"Parent directory: {parent_dir}\n"
            context += f"Directory contents: {', '.join(dir_contents)}\n"
            
            return context
        except Exception as e:
            self.log_debug(f"Error getting directory context: {str(e)}")
            return "Unable to determine directory context."
            
    def call_api_with_animation(self, api_function, **kwargs):
        """
        Call an API with a loading animation using Rich.
        
        Args:
            api_function: Function reference to the API call
            **kwargs: Keyword arguments to pass to the API function
            
        Returns:
            The API response
        """
        spinner_text = "Thinking..."
        spinner_style = "bold blue"
        
        # Show a spinner animation while waiting for the API response
        with console.status(f"[{spinner_style}]{spinner_text}[/{spinner_style}]", spinner="dots") as status:
            try:
                # Call the API function with the provided arguments
                response = api_function(**kwargs)
                return response
            except Exception as e:
                # Log any errors but don't display to user yet (caller will handle that)
                self.log_debug(f"API call error: {str(e)}")
                raise
    
    def is_shell_command(self, user_input: str) -> bool:
        """
        Determine if user input is likely a shell command that can be executed directly.
        This method is kept for backward compatibility but now uses the API-based approach.
        
        Args:
            user_input: User's input
            
        Returns:
            bool: True if the input appears to be a shell command, False otherwise
        """
        # Use the API-based classification instead of rule-based patterns
        intent = self.classify_user_intent(user_input)
        
        # Log the classification result
        self.log_debug(f"Shell command check via API: {intent['primary_intent']}")
        
        # Return true if the API classified this as a direct command
        return intent['primary_intent'] == 'direct_command'
    
    def _is_in_quotes(self, text: str, substring: str) -> bool:
        """
        Check if a substring appears only inside quotes in the text.
        
        Args:
            text: The full text string
            substring: The substring to check
            
        Returns:
            bool: True if the substring only appears inside quotes, False otherwise
        """
        in_single_quotes = False
        in_double_quotes = False
        
        for i in range(len(text)):
            # Toggle quote state
            if text[i] == "'" and not in_double_quotes:
                in_single_quotes = not in_single_quotes
            elif text[i] == '"' and not in_single_quotes:
                in_double_quotes = not in_double_quotes
                
            # Check if we're at the start of our substring
            if i <= len(text) - len(substring) and text[i:i+len(substring)] == substring:
                # If we're not in quotes, the substring is not exclusively in quotes
                if not in_single_quotes and not in_double_quotes:
                    return False
        
        # If we've found instances but they were all in quotes
        return True
    
    def refine_user_query(self, user_input: str) -> str:
        """
        Refine a user query to make it more precise for command generation.
        
        Args:
            user_input: Original user query
            
        Returns:
            str: Refined user query
        """
        if not self.refine_queries:
            return user_input
            
        self.log_debug(f"Refining user query: {user_input}")
        
        # Get directory context
        dir_context = self.get_directory_context()
        
        # Get conversation history for additional context
        conversation_history = self.logger.get_conversation_history()
        self.logger.log_system_message("Using conversation history for query refinement")
        
        # Create a system message for query refinement
        system_message = """You are an AI assistant that refines user queries into clear, specific instructions.
        Your task is to clarify ambiguous requests and add necessary context.
        Do NOT generate commands directly - just improve the query for clarity.
        Keep the refined query brief and focused.
        Consider the conversation history to understand the user's current context and needs.
        """
        
        # Create the messages for the API call
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Refine this query for a command-line assistant: '{user_input}'\n\nDirectory Information:\n{dir_context}\n\nRecent Conversation History:\n{conversation_history}"}
        ]
        
        try:
            # Call OpenAI API with the request
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.3
            )
            
            # Extract the refined query
            refined_query = response.choices[0].message.content.strip()
            self.log_debug(f"Refined query: {refined_query}")
            
            # If the refinement significantly changes the query, let the user know
            if len(refined_query) > len(user_input) * 1.5 or len(refined_query) < len(user_input) * 0.5:
                console.print(f"[dim]Refined query: {refined_query}[/dim]")
                
            return refined_query
            
        except Exception as e:
            # Log error but continue with original query
            self.log_debug(f"Error refining query: {str(e)}")
            return user_input
            
    def classify_user_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Unified function to classify user input intent using the API,
        replacing multiple rule-based detection methods.
        
        Args:
            user_input: User's input text
            
        Returns:
            Dict[str, Any]: Classification results with intent categories and confidence scores
        """
        self.log_debug(f"Classifying user intent: {user_input}")
        
        # Get context information
        dir_context = self.get_directory_context()
        conversation_history = self.logger.get_conversation_history()
        
        # Create a system message for intent classification
        system_message = """You are an AI assistant that classifies user input intent for a terminal assistant.
        Your task is to analyze user input and determine the type of request.
        
        Respond with a JSON object containing the following fields:
        - "primary_intent": One of the following categories:
          * "general_information" - General information question that doesn't require command execution
          * "content_creation" - Writing, reporting, documentation, analysis
          * "complex_task" - Task requiring multiple commands/steps
          * "context_request" - User asking about previous actions/commands
          * "direct_command" - A specific shell command
          * "simple_task" - Simple task requiring a single command
          
        - "confidence": Confidence score (0-1) for the primary intent
        - "dangerous_command": Boolean indicating if this might be a dangerous command (if applicable)
        - "risk_level": Number from 0-10 indicating potential risk level of the command (if applicable)
        - "risk_reasons": List of reasons why the command is risky (if applicable)
        - "requires_agent_mode": Boolean indicating if this task should use agent mode
        - "analysis": Brief explanation of why this classification was chosen
        
        When assessing command danger, consider:
        - Data deletion or modification risks
        - System-wide changes
        - Privilege escalation
        - Resource exhaustion
        - Security implications
        - Network and external access
        - Irreversible actions
        
        Base your classification on the specific request, directory context, and conversation history.
        """
        
        # Create the messages for the API call
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Classify this user input: '{user_input}'\n\nDirectory Information:\n{dir_context}\n\nRecent Conversation History:\n{conversation_history}"}
        ]
        
        try:
            # Call OpenAI API with the request
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            # Extract and parse the classification result
            classification = json.loads(response.choices[0].message.content)
            self.log_debug(f"Intent classification result: {classification}")
            
            # Log the classification decision
            self.logger.log_system_message(f"Intent classification: {classification['primary_intent']} (confidence: {classification['confidence']})")
            if 'analysis' in classification:
                self.logger.log_system_message(f"Analysis: {classification['analysis']}")
            
            # Log risk assessment if available
            if 'risk_level' in classification:
                self.logger.log_system_message(f"Risk assessment: Level {classification['risk_level']}/10")
                if 'risk_reasons' in classification and classification['risk_reasons']:
                    self.logger.log_system_message(f"Risk reasons: {', '.join(classification['risk_reasons'])}")
            
            return classification
            
        except Exception as e:
            # Log error and return a default classification
            error_msg = f"Error in intent classification: {str(e)}"
            self.log_debug(error_msg)
            self.logger.log_system_message(error_msg)
            
            # Return a conservative default
            return {
                "primary_intent": "simple_task",  # Default to simple task
                "confidence": 0.5,
                "dangerous_command": False,
                "requires_agent_mode": False,
                "analysis": "Classification failed, defaulting to simple task"
            }
    
    def extract_key_outputs(self, executed_steps: List[Dict]) -> List[str]:
        """
        Extract key findings or outputs from the executed steps.
        
        Args:
            executed_steps: List of executed steps with their outputs
            
        Returns:
            List[str]: Important outputs or findings
        """
        # First, collect all outputs that might contain useful information
        candidate_outputs = []
        
        for step in executed_steps:
            if step['success'] and step['output'].strip():
                # Prioritize execution phase outputs
                priority = 1 if step['phase'] == 'execution' else 2
                candidate_outputs.append({
                    "step": step['step_number'],
                    "output": step['output'].strip(),
                    "description": step['description'],
                    "priority": priority
                })
        
        # Simple heuristic: Outputs that are not too short, not too long
        filtered_outputs = []
        for output in candidate_outputs:
            lines = output['output'].split('\n')
            if 0 < len(lines) < 20:  # Not too many lines
                filtered_outputs.append(output)
        
        # Sort by priority (execution steps first) and limit to 5
        filtered_outputs.sort(key=lambda x: x['priority'])
        
        # Format the outputs for display
        key_outputs = []
        for output in filtered_outputs[:5]:
            # Try to summarize the output if it's more than 3 lines
            lines = output['output'].split('\n')
            if len(lines) > 3:
                summary = f"Step {output['step']} ({output['description']}): Found {len(lines)} items"
                # Add a sample of items if appropriate
                if len(lines) < 10:
                    sample = ", ".join(line.strip() for line in lines[:3])
                    summary += f" including {sample}..."
                key_outputs.append(summary)
            else:
                # Short outputs can be included directly
                key_outputs.append(f"Step {output['step']} ({output['description']}): {output['output']}")
        
        return key_outputs
    
    def handle_content_creation_task(self, task: str):
        """
        Handle content creation tasks like writing reports, articles, etc.
        
        Args:
            task: The content creation task description
        """
        self.logger.log_system_message(f"Creating content for task: {task}")
        console.print("[blue]Planning content structure...[/blue]")
        
        # Generate content outline first
        outline = self.generate_content_outline(task)
        
        # Display the outline to the user
        console.print("\n[bold blue]Content Outline:[/bold blue]")
        console.print(Markdown(outline))
        
        # Ask for confirmation before proceeding with full content generation
        if self.mode == "manual" and not Confirm.ask("Generate content based on this outline?", default=True):
            console.print("[blue]Content generation cancelled.[/blue]")
            return
        
        # Generate the full content
        console.print("\n[blue]Generating content... (this may take a few moments)[/blue]")
        content = self.generate_full_content(task, outline)
        
        # Display the content to the user
        console.print("\n[bold blue]Generated Content:[/bold blue]")
        console.print(Markdown(content))
        
        # Log the generated content
        self.logger.log_system_message("Content generated successfully")
        
        # Ask if user wants to save the content to a file
        if Confirm.ask("Would you like to save this content to a file?", default=True):
            filename = self.get_filename_for_content(task)
            self.save_content_to_file(content, filename)
    
    def generate_content_outline(self, task: str) -> str:
        """
        Generate an outline for content creation tasks.
        
        Args:
            task: Content creation task description
            
        Returns:
            str: Content outline in markdown format
        """
        system_message = """You are an AI assistant that creates detailed outlines for content.
        Create a comprehensive outline for the requested content.
        Format your outline in Markdown with clear sections and subsections.
        Be thorough but concise in your outline structure.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Create a detailed outline for the following content task: {task}"}
        ]
        
        try:
            # Use the animation helper for the API call
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.3
            )
            
            outline = response.choices[0].message.content
            self.logger.log_system_message("Content outline generated")
            return outline
            
        except Exception as e:
            error_msg = f"Error generating content outline: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(error_msg)
            return "# Error in Outline Generation\n\nFailed to create outline."
    
    def generate_full_content(self, task: str, outline: str) -> str:
        """
        Generate full content based on task and outline.
        
        Args:
            task: Content creation task description
            outline: Content outline
            
        Returns:
            str: Generated content in markdown format
        """
        system_message = """You are an AI assistant that creates detailed, well-researched content.
        Create comprehensive content based on the provided task and outline.
        Make sure to follow the outline structure but expand with detailed information.
        Format your response using Markdown for better readability.
        Include proper headings, subheadings, bullet points, and formatting.
        Aim to be informative, accurate, and well-structured.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Create detailed content for the following task: {task}\n\nUse this outline as the structure:\n\n{outline}"}
        ]
        
        try:
            # Use the animation helper for the API call
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            self.logger.log_system_message("Full content generated")
            return content
            
        except Exception as e:
            error_msg = f"Error generating full content: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(error_msg)
            return "# Error in Content Generation\n\nFailed to create content."
    
    def get_filename_for_content(self, task: str) -> str:
        """
        Generate a suitable filename for the content.
        
        Args:
            task: The content task
            
        Returns:
            str: A suitable filename
        """
        # Sanitize task to create filename
        sanitized = re.sub(r'[^\w\s-]', '', task.lower())
        sanitized = re.sub(r'[\s-]+', '_', sanitized)
        base_filename = sanitized[:50]  # Limit length
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_filename}_{timestamp}.md"
        
        # Allow user to modify the filename
        user_filename = Prompt.ask("[bold]Enter filename to save content[/bold]", default=filename)
        
        # Ensure .md extension
        if not user_filename.endswith('.md'):
            user_filename += '.md'
            
        return user_filename
    
    def save_content_to_file(self, content: str, filename: str):
        """
        Save generated content to a file.
        
        Args:
            content: The content to save
            filename: Filename to save to
        """
        try:
            # Create full path
            file_path = os.path.join(os.getcwd(), filename)
            
            # Write content to file
            with open(file_path, 'w') as f:
                f.write(content)
                
            console.print(f"[bold green]Content saved to [/bold green][bold cyan]{file_path}[/bold cyan]")
            self.logger.log_system_message(f"Content saved to file: {file_path}")
            
            # Execute a command to show the file was created
            self.execute_command(f"ls -la {shlex.quote(file_path)}")
            
        except Exception as e:
            error_msg = f"Error saving content to file: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(error_msg)
    
    def generate_plan(self, task: str) -> Dict:
        """
        Generate a step-by-step plan for a complex task.
        
        Args:
            task: Complex task description
            
        Returns:
            Dict: Plan in JSON format with an array of steps
        """
        # Get step-by-step plan from OpenAI
        system_message = """You are an AI assistant that breaks down complex tasks into a series of shell commands for macOS.
        Your approach should mirror human-like reasoning by ALWAYS following these phases:

        PHASE 1: ANALYSIS & DISCOVERY
        - List and examine what exists in the environment
        - Categorize what you find (file types, structures, patterns)
        - Understand the current state before making changes
        
        PHASE 2: PLANNING
        - Based on your analysis, develop a thoughtful approach
        - Consider multiple ways to accomplish the goal
        - Choose the most appropriate method based on context

        PHASE 3: EXECUTION
        - Only after thorough analysis, create commands to accomplish the goal
        - Include verification steps to confirm actions worked as expected
        
        Create a step-by-step plan with specific shell commands to accomplish the user's goal.
        Format your response as JSON with an array of steps, each with:
        - description: Brief description of what this step does
        - command: The exact shell command to run
        - critical: Boolean indicating if this step is critical (failure should stop execution)
        - phase: String indicating which phase this step belongs to ("analysis", "planning", or "execution")
        
        IMPORTANT GUIDELINES FOR COMMAND EXECUTION:
        
        1. DIRECTORY PERSISTENCE: Each command is executed in a separate subprocess, so directory changes with 'cd' DO NOT persist between steps.
           When working with directories, choose ONE of these approaches:
           - Use absolute paths in commands
           - Combine 'cd' and the actual command in the same step with '&&' (e.g., "cd /path && command")
           - Use relative paths from the initial directory

        2. When writing files to a new directory, either:
           - Use the full path: "echo content > /path/to/dir/file.txt"
           - Combine commands: "cd /path/to/dir && echo content > file.txt"
           
        3. Avoid unnecessary 'cd' commands when you can specify the full path directly
        
        4. For multi-part operations in the same directory, consider combining them with '&&' in a single step
        
        EXAMPLE OF A WELL-STRUCTURED PLAN:
        ```
        {
          "steps": [
            {
              "description": "Analyze what files and directories exist in the current location",
              "command": "find . -type f -o -type d | sort",
              "critical": true,
              "phase": "analysis"
            },
            {
              "description": "Examine file types to understand what we're working with",
              "command": "find . -type f -exec file {} \\;",
              "critical": true,
              "phase": "analysis"
            },
            {
              "description": "Check for large files that might need special handling",
              "command": "find . -type f -size +10M -exec ls -lh {} \\;",
              "critical": false,
              "phase": "analysis"
            },
            {
              "description": "Based on analysis, create a directory structure for organization",
              "command": "mkdir -p media/images documents/text data/csv",
              "critical": true,
              "phase": "planning"
            },
            {
              "description": "Move image files to the images directory",
              "command": "find . -maxdepth 1 -type f -name '*.jpg' -o -name '*.png' -exec mv {} ./media/images/ \\;",
              "critical": true,
              "phase": "execution"
            },
            {
              "description": "Verify files were moved successfully",
              "command": "ls -la ./media/images/",
              "critical": false,
              "phase": "execution"
            }
          ]
        }
        ```
        """
        
        # Get directory context
        dir_context = self.get_directory_context()
        
        user_content = f"""Task: {task}

Directory Information:
{dir_context}

IMPORTANT NOTE: Remember that each command runs separately, so directory changes with 'cd' won't persist across steps. 
Use absolute paths, combined commands with &&, or ensure your paths account for this limitation.

REMEMBER: Follow the human-like reasoning process by first analyzing what exists, then planning, and finally executing the plan."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        try:
            # Show thinking animation while creating the plan
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            return plan
            
        except Exception as e:
            console.print(f"[bold red]Error generating plan: {str(e)}[/bold red]")
            if self.debug:
                import traceback
                console.print(traceback.format_exc())
            return None
    
    def verify_and_refine_plan(self, task: str, initial_plan: Dict) -> Dict:
        """
        Verify and refine a plan through multiple iterations.
        
        Args:
            task: Original task description
            initial_plan: Initial plan generated
            
        Returns:
            Dict: Final verified and refined plan
        """
        current_plan = initial_plan
        iterations = 0
        
        console.print("[blue]Verifying and refining the plan...[/blue]")
        
        while iterations < self.max_plan_iterations:
            # Create a progress message
            if iterations > 0:
                console.print(f"[dim]Plan refinement iteration {iterations+1}/{self.max_plan_iterations}[/dim]")
            
            # Verify the current plan
            verification_result = self.verify_plan(task, current_plan)
            
            # If the plan is verified as good, break the loop
            if verification_result.get('is_good', False):
                console.print("[green]Plan verification completed: Plan is sound.[/green]")
                break
                
            # If refinements are suggested, update the plan
            if 'refined_plan' in verification_result and verification_result['refined_plan']:
                # Check if the refined plan is different from the current one
                if verification_result['refined_plan'] != current_plan:
                    console.print("[blue]Refining plan based on verification feedback...[/blue]")
                    current_plan = verification_result['refined_plan']
                else:
                    # No changes in the plan, break the loop
                    console.print("[green]Plan verification completed: No further refinements needed.[/green]")
                    break
            else:
                # No refinements suggested, break the loop
                console.print("[yellow]Plan verification completed: No refinements suggested.[/yellow]")
                break
                
            iterations += 1
            
        if iterations >= self.max_plan_iterations:
            console.print("[yellow]Reached maximum plan refinement iterations.[/yellow]")
            
        return current_plan
    
    def verify_plan(self, task: str, plan: Dict) -> Dict:
        """
        Verify a plan using OpenAI API.
        
        Args:
            task: Original task description
            plan: Plan to verify
            
        Returns:
            Dict: Verification result with is_good flag and possibly a refined_plan
        """
        # Get directory context
        dir_context = self.get_directory_context()
        
        # Get conversation history
        conv_history = self.logger.get_conversation_history()
        
        # Create a system message for plan verification
        system_message = """You are an AI assistant that verifies the soundness of shell command execution plans.
        Analyze the given plan and task to determine if:
        1. The plan accomplishes the task completely
        2. The commands are correct and efficient
        3. The steps are in the right order
        4. There are no missing steps
        5. There are no unnecessary steps
        
        IMPORTANT: The plan must follow a human-like reasoning process with these phases:
        
        PHASE 1: ANALYSIS & DISCOVERY
        - At least 2-3 steps should be dedicated to analyzing what exists before taking action
        - The plan should include commands to list and examine files/directories
        - There should be steps to categorize or understand the content (file types, sizes, etc.)
        
        PHASE 2: PLANNING
        - Based on the analysis, the plan should include steps for organizing the approach
        - This may include creating directory structures or determining what needs to be done
        
        PHASE 3: EXECUTION
        - Only after thorough analysis and planning should the actual task be executed
        - Verification steps should be included to confirm actions worked as expected
        
        CRITICAL ISSUES TO CHECK:
        
        - Directory persistence: When the plan includes a 'cd' command, subsequent commands will NOT inherit the directory change when executed separately.
          This is because each command runs in its own subprocess. Fix this by either:
          a) Combining steps with directory changes using && (e.g., "cd dir && command")
          b) Using absolute paths in subsequent commands
          c) Prepending the target directory to all file operations (e.g., "echo content > dir/file" instead of "cd dir" then "echo content > file")
        
        - Command dependencies: Ensure that commands which depend on previous commands' outputs are properly sequenced
        
        - Error handling: Consider adding checks to verify critical steps succeeded before proceeding
        
        - Permission issues: Check if commands might require permissions (e.g., sudo) or create files in protected directories
        
        - File overwriting: Be cautious about commands that might overwrite existing files without confirmation
        
        If improvements are needed, provide a refined plan in the same format.
        Format your response as JSON with:
        - is_good: Boolean indicating if the plan is sound (true) or needs refinement (false)
        - feedback: Brief explanation of your assessment, particularly highlighting issues like directory persistence or missing analysis steps
        - refined_plan: The improved plan if is_good is false, otherwise null
        """
        
        # Convert plan to a formatted string for readability
        plan_text = json.dumps(plan, indent=2)
        
        # Create the user message
        user_content = f"""Task: {task}

Proposed Plan:
{plan_text}

Directory Information:
{dir_context}

Recent Conversation History:
{conv_history}

Please verify if this plan is sound and complete for accomplishing the task. 
The plan MUST follow a human-like reasoning process with proper analysis steps before taking action.
Pay special attention to directory persistence issues - if any step changes directory with 'cd', subsequent commands need to account for this by either using absolute paths or combining commands with &&."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        try:
            # Call the API with animation
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Display feedback
            if 'feedback' in result:
                console.print(f"[dim]Verification feedback: {result['feedback']}[/dim]")
                self.logger.log_system_message(f"Plan verification feedback: {result['feedback']}")
                
            return result
            
        except Exception as e:
            error_msg = f"Error verifying plan: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            self.logger.log_system_message(f"Error: {error_msg}")
            # Return a default result indicating the plan is good to avoid blocking execution
            return {"is_good": True, "feedback": "Verification failed, proceeding with original plan."}
    
    def get_directory_context(self) -> str:
        """
        Get context information about the current directory environment.
        
        Returns:
            str: Context information about current directory and environment
        """
        try:
            # Get current directory path
            current_dir = os.getcwd()
            
            # Get directory contents (files and folders)
            dir_contents = os.listdir(current_dir)
            # Limit the list to avoid excessive information
            if len(dir_contents) > 10:
                dir_contents = dir_contents[:10] + ["..."]
            
            # Get parent directory
            parent_dir = os.path.dirname(current_dir)
            
            # Construct the context information
            context = f"Current directory: {current_dir}\n"
            context += f"Parent directory: {parent_dir}\n"
            context += f"Directory contents: {', '.join(dir_contents)}\n"
            
            return context
        except Exception as e:
            self.log_debug(f"Error getting directory context: {str(e)}")
            return "Unable to determine directory context."
            
    def call_api_with_animation(self, api_function, **kwargs):
        """
        Call an API with a loading animation using Rich.
        
        Args:
            api_function: Function reference to the API call
            **kwargs: Keyword arguments to pass to the API function
            
        Returns:
            The API response
        """
        spinner_text = "Thinking..."
        spinner_style = "bold blue"
        
        # Show a spinner animation while waiting for the API response
        with console.status(f"[{spinner_style}]{spinner_text}[/{spinner_style}]", spinner="dots") as status:
            try:
                # Call the API function with the provided arguments
                response = api_function(**kwargs)
                return response
            except Exception as e:
                # Log any errors but don't display to user yet (caller will handle that)
                self.log_debug(f"API call error: {str(e)}")
                raise
    
    def is_shell_command(self, user_input: str) -> bool:
        """
        Determine if user input is likely a shell command that can be executed directly.
        This method is kept for backward compatibility but now uses the API-based approach.
        
        Args:
            user_input: User's input
            
        Returns:
            bool: True if the input appears to be a shell command, False otherwise
        """
        # Use the API-based classification instead of rule-based patterns
        intent = self.classify_user_intent(user_input)
        
        # Log the classification result
        self.log_debug(f"Shell command check via API: {intent['primary_intent']}")
        
        # Return true if the API classified this as a direct command
        return intent['primary_intent'] == 'direct_command'
    
    def _is_in_quotes(self, text: str, substring: str) -> bool:
        """
        Check if a substring appears only inside quotes in the text.
        
        Args:
            text: The full text string
            substring: The substring to check
            
        Returns:
            bool: True if the substring only appears inside quotes, False otherwise
        """
        in_single_quotes = False
        in_double_quotes = False
        
        for i in range(len(text)):
            # Toggle quote state
            if text[i] == "'" and not in_double_quotes:
                in_single_quotes = not in_single_quotes
            elif text[i] == '"' and not in_single_quotes:
                in_double_quotes = not in_double_quotes
                
            # Check if we're at the start of our substring
            if i <= len(text) - len(substring) and text[i:i+len(substring)] == substring:
                # If we're not in quotes, the substring is not exclusively in quotes
                if not in_single_quotes and not in_double_quotes:
                    return False
        
        # If we've found instances but they were all in quotes
        return True
    
    def refine_user_query(self, user_input: str) -> str:
        """
        Refine a user query to make it more precise for command generation.
        
        Args:
            user_input: Original user query
            
        Returns:
            str: Refined user query
        """
        if not self.refine_queries:
            return user_input
            
        self.log_debug(f"Refining user query: {user_input}")
        
        # Get directory context
        dir_context = self.get_directory_context()
        
        # Get conversation history for additional context
        conversation_history = self.logger.get_conversation_history()
        self.logger.log_system_message("Using conversation history for query refinement")
        
        # Create a system message for query refinement
        system_message = """You are an AI assistant that refines user queries into clear, specific instructions.
        Your task is to clarify ambiguous requests and add necessary context.
        Do NOT generate commands directly - just improve the query for clarity.
        Keep the refined query brief and focused.
        Consider the conversation history to understand the user's current context and needs.
        """
        
        # Create the messages for the API call
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Refine this query for a command-line assistant: '{user_input}'\n\nDirectory Information:\n{dir_context}\n\nRecent Conversation History:\n{conversation_history}"}
        ]
        
        try:
            # Call OpenAI API with the request
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                temperature=0.3
            )
            
            # Extract the refined query
            refined_query = response.choices[0].message.content.strip()
            self.log_debug(f"Refined query: {refined_query}")
            
            # If the refinement significantly changes the query, let the user know
            if len(refined_query) > len(user_input) * 1.5 or len(refined_query) < len(user_input) * 0.5:
                console.print(f"[dim]Refined query: {refined_query}[/dim]")
                
            return refined_query
            
        except Exception as e:
            # Log error but continue with original query
            self.log_debug(f"Error refining query: {str(e)}")
            return user_input
            
    def classify_user_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Unified function to classify user input intent using the API,
        replacing multiple rule-based detection methods.
        
        Args:
            user_input: User's input text
            
        Returns:
            Dict[str, Any]: Classification results with intent categories and confidence scores
        """
        self.log_debug(f"Classifying user intent: {user_input}")
        
        # Get context information
        dir_context = self.get_directory_context()
        conversation_history = self.logger.get_conversation_history()
        
        # Create a system message for intent classification
        system_message = """You are an AI assistant that classifies user input intent for a terminal assistant.
        Your task is to analyze user input and determine the type of request.
        
        Respond with a JSON object containing the following fields:
        - "primary_intent": One of the following categories:
          * "general_information" - General information question that doesn't require command execution
          * "content_creation" - Writing, reporting, documentation, analysis
          * "complex_task" - Task requiring multiple commands/steps
          * "context_request" - User asking about previous actions/commands
          * "direct_command" - A specific shell command
          * "simple_task" - Simple task requiring a single command
          
        - "confidence": Confidence score (0-1) for the primary intent
        - "dangerous_command": Boolean indicating if this might be a dangerous command (if applicable)
        - "risk_level": Number from 0-10 indicating potential risk level of the command (if applicable)
        - "risk_reasons": List of reasons why the command is risky (if applicable)
        - "requires_agent_mode": Boolean indicating if this task should use agent mode
        - "analysis": Brief explanation of why this classification was chosen
        
        When assessing command danger, consider:
        - Data deletion or modification risks
        - System-wide changes
        - Privilege escalation
        - Resource exhaustion
        - Security implications
        - Network and external access
        - Irreversible actions
        
        Base your classification on the specific request, directory context, and conversation history.
        """
        
        # Create the messages for the API call
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Classify this user input: '{user_input}'\n\nDirectory Information:\n{dir_context}\n\nRecent Conversation History:\n{conversation_history}"}
        ]
        
        try:
            # Call OpenAI API with the request
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            # Extract and parse the classification result
            classification = json.loads(response.choices[0].message.content)
            self.log_debug(f"Intent classification result: {classification}")
            
            # Log the classification decision
            self.logger.log_system_message(f"Intent classification: {classification['primary_intent']} (confidence: {classification['confidence']})")
            if 'analysis' in classification:
                self.logger.log_system_message(f"Analysis: {classification['analysis']}")
            
            # Log risk assessment if available
            if 'risk_level' in classification:
                self.logger.log_system_message(f"Risk assessment: Level {classification['risk_level']}/10")
                if 'risk_reasons' in classification and classification['risk_reasons']:
                    self.logger.log_system_message(f"Risk reasons: {', '.join(classification['risk_reasons'])}")
            
            return classification
            
        except Exception as e:
            # Log error and return a default classification
            error_msg = f"Error in intent classification: {str(e)}"
            self.log_debug(error_msg)
            self.logger.log_system_message(error_msg)
            
            # Return a conservative default
            return {
                "primary_intent": "simple_task",  # Default to simple task
                "confidence": 0.5,
                "dangerous_command": False,
                "requires_agent_mode": False,
                "analysis": "Classification failed, defaulting to simple task"
            }
    
    def extract_key_outputs(self, executed_steps: List[Dict]) -> List[str]:
        """
        Extract key findings or outputs from the executed steps.
        
        Args:
            executed_steps: List of executed steps with their outputs
            
        Returns:
            List[str]: Important outputs or findings
        """
        # First, collect all outputs that might contain useful information
        candidate_outputs = []
        
        for step in executed_steps:
            if step['success'] and step['output'].strip():
                # Prioritize execution phase outputs
                priority = 1 if step['phase'] == 'execution' else 2
                candidate_outputs.append({
                    "step": step['step_number'],
                    "output": step['output'].strip(),
                    "description": step['description'],
                    "priority": priority
                })
        
        # Simple heuristic: Outputs that are not too short, not too long
        filtered_outputs = []
        for output in candidate_outputs:
            lines = output['output'].split('\n')
            if 0 < len(lines) < 20:  # Not too many lines
                filtered_outputs.append(output)
        
        # Sort by priority (execution steps first) and limit to 5
        filtered_outputs.sort(key=lambda x: x['priority'])
        
        # Format the outputs for display
        key_outputs = []
        for output in filtered_outputs[:5]:
            # Try to summarize the output if it's more than 3 lines
            lines = output['output'].split('\n')
            if len(lines) > 3:
                summary = f"Step {output['step']} ({output['description']}): Found {len(lines)} items"
                # Add a sample of items if appropriate
                if len(lines) < 10:
                    sample = ", ".join(line.strip() for line in lines[:3])
                    summary += f" including {sample}..."
                key_outputs.append(summary)
            else:
                # Short outputs can be included directly
                key_outputs.append(f"Step {output['step']} ({output['description']}): {output['output']}")
        
        return key_outputs
    
    def generate_high_level_plan(self, task: str) -> Dict:
        """
        Generate a high-level plan without specific commands.
        
        Args:
            task: Complex task description
            
        Returns:
            Dict: High-level plan with step descriptions but no commands
        """
        system_message = """You are an AI assistant that breaks down complex tasks into a series of high-level steps.
        Your approach should mirror human-like reasoning by ALWAYS following these phases:

        PHASE 1: ANALYSIS & DISCOVERY
        - List and examine what exists in the environment
        - Categorize what you find (file types, structures, patterns)
        - Understand the current state before making changes
        
        PHASE 2: PLANNING
        - Based on your analysis, develop a thoughtful approach
        - Consider multiple ways to accomplish the goal
        - Choose the most appropriate method based on context

        PHASE 3: EXECUTION
        - Only after thorough analysis, create high-level steps to accomplish the goal
        - Include verification steps to confirm actions worked as expected
        
        Create a step-by-step plan with clear descriptions but DO NOT include specific shell commands.
        Format your response as JSON with an array of steps, each with:
        - description: Clear description of what this step should accomplish
        - expected_outcome: What should happen if this step succeeds
        - critical: Boolean indicating if this step is critical (failure should stop execution)
        - phase: String indicating which phase this step belongs to ("analysis", "planning", or "execution")
        
        EXAMPLE OF A WELL-STRUCTURED PLAN:
        ```
        {
          "steps": [
            {
              "description": "Analyze what files and directories exist in the current location",
              "expected_outcome": "Understanding of the current file structure and contents",
              "critical": true,
              "phase": "analysis"
            },
            {
              "description": "Examine file types to understand what we're working with",
              "expected_outcome": "Categorization of files by type",
              "critical": true,
              "phase": "analysis"
            },
            {
              "description": "Create a directory structure for organization",
              "expected_outcome": "Creation of needed directories for organization",
              "critical": true,
              "phase": "planning"
            },
            {
              "description": "Move image files to the images directory",
              "expected_outcome": "Image files relocated to the appropriate directory",
              "critical": true,
              "phase": "execution"
            },
            {
              "description": "Verify files were moved successfully",
              "expected_outcome": "Confirmation that files exist in the new location",
              "critical": false,
              "phase": "execution"
            }
          ]
        }
        ```
        """
        
        # Get directory context
        dir_context = self.get_directory_context()
        
        user_content = f"""Task: {task}

Directory Information:
{dir_context}

IMPORTANT NOTE: Create a high-level plan without specific commands. Focus on what each step should accomplish.

REMEMBER: Follow the human-like reasoning process by first analyzing what exists, then planning, and finally executing the plan."""
        
        try:
            # Show thinking animation while creating the plan
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            return plan
            
        except Exception as e:
            console.print(f"[bold red]Error generating high-level plan: {str(e)}[/bold red]")
            if self.debug:
                import traceback
                console.print(traceback.format_exc())
            return None
    
    def generate_step_command(self, task: str, step: Dict, previous_steps: List[Dict]) -> str:
        """
        Generate a specific command for a high-level step.
        
        Args:
            task: Original task description
            step: Current step to generate a command for
            previous_steps: List of previously executed steps
            
        Returns:
            str: Generated command for this step
        """
        # Get the phase for this step
        phase = step.get('phase', 'unknown')
        
        # Customize system message based on the phase
        phase_guidance = ""
        if phase == 'analysis':
            phase_guidance = """
            IMPORTANT: This is an ANALYSIS phase step.
            - Use ONLY read-only commands (ls, find, grep, cat, etc.)
            - DO NOT generate commands that modify the filesystem (touch, mkdir, cp, mv, rm, etc.)
            - Focus on gathering information and understanding the system state
            """
        elif phase == 'planning':
            phase_guidance = """
            IMPORTANT: This is a PLANNING phase step.
            - Focus on preparing for execution without making permanent changes
            - DO NOT generate commands that delete data (rm, rmdir, etc.)
            - DO NOT generate commands that move files (mv, rename, etc.)
            - You may use commands that create new files/directories or display information
            """
        elif phase == 'execution':
            phase_guidance = """
            IMPORTANT: This is an EXECUTION phase step.
            - You may generate commands that modify the filesystem as needed
            - Ensure commands are precise and achieve the step's purpose
            """
        
        system_message = """You are an AI assistant that generates specific shell commands for macOS based on a high-level step description.
        Given a task, the current step, and information about previous steps, generate the exact shell command that should be executed.
        
        IMPORTANT GUIDELINES FOR COMMAND GENERATION:
        
        1. DIRECTORY PERSISTENCE: Each command is executed in a separate subprocess, so directory changes with 'cd' DO NOT persist between steps.
           When working with directories, choose ONE of these approaches:
           - Use absolute paths in commands
           - Combine 'cd' and the actual command in the same step with '&&' (e.g., "cd /path && command")
           - Use relative paths from the initial directory

        2. When writing files to a new directory, either:
           - Use the full path: "echo content > /path/to/dir/file.txt"
           - Combine commands: "cd /path/to/dir && echo content > file.txt"
           
        3. Avoid unnecessary 'cd' commands when you can specify the full path directly
        
        4. Generate ONLY the shell command, without any prefixes, language indicators, or formatting.
           DO NOT include language indicators like 'shell:', 'bash:', or any formatting marks.
           NEVER prefix the command with words like 'shell', 'bash', 'zsh'.
           JUST output the raw command that should be executed, nothing else.
        """
        
        # Add the phase-specific guidance
        system_message += phase_guidance
        
        # Get directory context
        dir_context = self.get_directory_context()
        
        # Format information about previous steps
        previous_steps_info = ""
        for i, prev_step in enumerate(previous_steps, 1):
            previous_steps_info += f"Step {i}: {prev_step['description']}\n"
            if 'expected_outcome' in prev_step:
                previous_steps_info += f"Expected outcome: {prev_step['expected_outcome']}\n"
            previous_steps_info += "\n"
        
        user_content = f"""Task: {task}

Current Directory Information:
{dir_context}

Previous Steps:
{previous_steps_info}

Current Step to Generate Command For:
Description: {step['description']}
Expected Outcome: {step.get('expected_outcome', 'No expected outcome provided')}
Phase: {phase}

Generate ONLY the shell command for this step. DO NOT include any language indicators, explanations, or formatting.
DO NOT prefix with 'shell', 'bash', or any other word. Just provide the raw command."""
        
        try:
            # Generate the command
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2
            )
            
            command = response.choices[0].message.content.strip()
            
            # Remove any markdown formatting if present
            if command.startswith("```") and command.endswith("```"):
                command = command[3:-3].strip()
            elif command.startswith("`") and command.endswith("`"):
                command = command[1:-1].strip()
                
            # Remove common language prefixes that might cause errors
            command = re.sub(r'^(?:shell|bash|zsh)[:]\s*', '', command)
            command = re.sub(r'^(?:shell|bash|zsh)\s+', '', command)
            
            # Log the cleaned command
            self.log_debug(f"Generated step command (after cleanup): {command}")
            
            # Validate the command for the current phase
            validation = self.validate_command_for_phase(command, phase)
            if not validation["valid"]:
                self.log_debug(f"Command validation failed: {validation['message']}")
                
                # If we're in debug mode, show the validation warning
                if self.debug:
                    console.print(f"[bold yellow]Warning: {validation['message']}[/bold yellow]")
                
                # In manual mode or if in planning/analysis phase where we want to be safe,
                # replace the command with an echo warning
                if self.mode == "manual" or phase in ['analysis', 'planning']:
                    command = f"echo 'WARNING: Generated command ({validation['command_type']}) is not appropriate for {phase} phase. Command was: {command}'"
                
            return command
            
        except Exception as e:
            console.print(f"[bold red]Error generating command: {str(e)}[/bold red]")
            if self.debug:
                import traceback
                console.print(traceback.format_exc())
            return f"echo 'Error generating command: {str(e)}'"
    
    def verify_step_success(self, step: Dict, command: str, output: str, execution_success: bool) -> Dict:
        """
        Verify if a step achieved its intended purpose beyond just execution success.
        
        Args:
            step: Step information
            command: The command that was executed
            output: Output from command execution
            execution_success: Whether the command executed without errors
            
        Returns:
            Dict: Verification result with success boolean and message
        """
        if not execution_success:
            return {
                "success": False,
                "message": "Command execution failed with errors."
            }
            
        # First validate if the command was appropriate for the phase
        phase = step.get('phase', 'unknown')
        validation = self.validate_command_for_phase(command, phase)
        
        # If the command was not valid for this phase, mark the step as failed
        if not validation["valid"]:
            return {
                "success": False,
                "message": f"Command ({validation['command_type']}) is not appropriate for {phase} phase. {validation['message']}"
            }
            
        # Special case for file listing commands - if they executed successfully, consider it a success
        if (command.strip().startswith("find") or command.strip().startswith("ls") or 
            "ls -" in command or "find ." in command):
            # For file listing commands, success means it ran without error, even if it found few or no files
            return {
                "success": True,
                "message": f"Command executed successfully. {'Files were found.' if output.strip() else 'No files matched the criteria, which is a valid outcome.'}"
            }
            
        system_message = """You are an AI assistant that verifies if a shell command achieved its intended purpose.
        Given a step description, expected outcome, the command that was executed, and its output,
        determine if the step was successful in achieving its intended purpose.
        
        Be critical but fair in your assessment. Don't just check if the command ran without errors,
        but verify if it actually accomplished what it was supposed to do based on the expected outcome.
        
        IMPORTANT GUIDELINES:
        1. For file listing commands, consider it a success if the command executed without errors,
           even if it found no files or just a few files.
        2. For search commands, finding no matches can be a valid result - it means the search was 
           performed but no items matched the criteria.
        3. Be realistic about success criteria - don't expect unreasonable outcomes.
        4. Pay special attention to the phase of the step:
           - Analysis phase: Should focus on information gathering without modifying the system
           - Planning phase: Should focus on preparation without destructive actions
           - Execution phase: Where actual modifications and changes should occur
        """
        
        user_content = f"""Step Description: {step['description']}
Expected Outcome: {step.get('expected_outcome', 'No expected outcome provided')}
Phase: {phase}
Command Executed: {command}
Command Output: 
{output}

Did this step achieve its intended purpose? Provide a boolean (true/false) and a brief explanation.

Remember:
- If this is a file listing or search command, finding few or no results is still success if the command executed properly.
- The expected outcome should be interpreted reasonably - don't be overly demanding.
- For phases:
  * Analysis phase steps should focus on information gathering, not system modification
  * Planning phase steps should not include destructive operations
  * Execution phase is where changes and modifications should occur
"""
        
        try:
            # Verify the step success
            response = openai.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1
            )
            
            verification_text = response.choices[0].message.content.lower()
            
            # Parse the verification response
            success = "true" in verification_text[:20] or "yes" in verification_text[:20]
            
            # Extract the explanation
            message = verification_text
            if "true" in verification_text[:20]:
                message = verification_text.split("true", 1)[1].strip()
            elif "false" in verification_text[:20]:
                message = verification_text.split("false", 1)[1].strip()
            elif "yes" in verification_text[:20]:
                message = verification_text.split("yes", 1)[1].strip()
            elif "no" in verification_text[:20]:
                message = verification_text.split("no", 1)[1].strip()
                
            # Clean up the message
            if message.startswith(".") or message.startswith(":") or message.startswith(","):
                message = message[1:].strip()
                
            return {
                "success": success,
                "message": message
            }
            
        except Exception as e:
            console.print(f"[bold red]Error verifying step: {str(e)}[/bold red]")
            if self.debug:
                import traceback
                console.print(traceback.format_exc())
            
            # More conservative approach: If we can't verify with AI, rely on command type validation
            # If command was not valid for this phase, we consider it failed
            if not validation["valid"]:
                return {
                    "success": False,
                    "message": f"Could not verify with AI, but command type ({validation['command_type']}) is not appropriate for {phase} phase."
                }
                
            return {
                "success": execution_success,  # Fall back to execution success
                "message": f"Could not verify due to error: {str(e)}"
            }
    
    def fix_failed_step(self, step: Dict, failed_command: str, output: str, verification: Dict) -> bool:
        """
        Try to fix a failed step and retry.
        
        Args:
            step: Step information
            failed_command: The command that failed
            output: Output from the failed command
            verification: The verification result
            
        Returns:
            bool: Whether the step was fixed successfully
        """
        console.print("[blue]Analyzing the failure and generating a fix...[/blue]")
        
        # Get current directory state to check if things have changed
        current_dir_state = self.get_directory_context()
        
        system_message = """You are an AI assistant that fixes failed shell commands.
        Given a step description, the command that failed, the error output,
        and the current directory state, analyze what went wrong and generate a new command that should work correctly.
        
        IMPORTANT GUIDELINES:
        1. If the failure was because the command wasn't appropriate for the step's phase, suggest a command that IS appropriate.
        2. Always check the current directory state to see if previous commands have already changed the system.
        3. If files or directories mentioned in the error no longer exist, DO NOT try to access them again.
        4. If the task has already been partially completed, only fix the remaining parts.
        5. Be especially careful with destructive commands - don't suggest them if the step's phase is 'analysis' or 'planning'.
        
        For each phase, ensure the command type is appropriate:
        - Analysis phase: Use only read-only commands (ls, find, grep, etc.)
        - Planning phase: Can create but not modify/delete (echo, mkdir, touch, etc.)
        - Execution phase: Can perform any operation needed
        """
        
        user_content = f"""Task Step: {step['description']}
                        Expected Outcome: {step.get('expected_outcome', 'No expected outcome provided')}
                        Phase: {step.get('phase', 'unknown')}
                        Failed Command: {failed_command}
                        Error/Output: 
                        {output}
                        Verification Result: {verification['message']}

                        Current Directory State:
                        {current_dir_state}

                        Please analyze what went wrong and generate a new command that will achieve the intended purpose.
                        Remember to consider the current state of the system - if files were already moved or deleted, don't try to access them again.
                        Also, ensure the command is appropriate for the step's phase ({step.get('phase', 'unknown')}).
                        """
        
        try:
            # Generate a fix
            response = self.call_api_with_animation(
                openai.chat.completions.create,
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2
            )
            
            analysis_and_fix = response.choices[0].message.content
            
            # Extract just the command portion if there's an explanation
            command_lines = []
            capture = False
            for line in analysis_and_fix.split('\n'):
                line = line.strip()
                if line.startswith('```') and not capture:
                    capture = True
                    continue
                elif line.startswith('```') and capture:
                    capture = False
                    continue
                elif capture:
                    command_lines.append(line)
                elif line.startswith('`') and line.endswith('`'):
                    command_lines.append(line[1:-1])
            
            # If we found code blocks, use those
            if command_lines:
                fixed_command = '\n'.join(command_lines)
            else:
                # Otherwise, use the last non-empty line as a fallback
                lines = [line for line in analysis_and_fix.split('\n') if line.strip()]
                fixed_command = lines[-1] if lines else analysis_and_fix
            
            # Display the analysis and the fix
            console.print("[bold blue]Analysis and Fix:[/bold blue]")
            console.print(analysis_and_fix)
            console.print("\n[bold blue]Executing fixed command:[/bold blue]")
            console.print(Syntax(fixed_command, "bash", theme="monokai", line_numbers=False))
            
            # First validate if the fixed command is appropriate for this phase
            phase = step.get('phase', 'unknown')
            validation = self.validate_command_for_phase(fixed_command, phase)
            
            # If the fixed command is still not valid for this phase, warn the user
            if not validation["valid"]:
                console.print(f"[bold red]Warning: The fixed command ({validation['command_type']}) is still not appropriate for {phase} phase.[/bold red]")
                
                # If in manual mode, ask if the user wants to proceed anyway
                if self.mode == "manual":
                    if not Confirm.ask("The command is not appropriate for this phase. Continue anyway?", default=False):
                        console.print("[yellow]Fix attempt cancelled.[/yellow]")
                        return False
                else:
                    # In autonomous mode, don't execute inappropriate commands
                    console.print("[yellow]Fix attempt failed: Generated command is not appropriate for this phase.[/yellow]")
                    return False
            
            # Check if the fixed command is dangerous
            danger_assessment = self.check_command_danger_with_ai(fixed_command)
            
            # Execute the fixed command
            if self.mode == "manual" or danger_assessment:
                output, success = self.execute_with_confirmation(fixed_command, "Fixed command", is_dangerous=danger_assessment)
            else:
                output, success = self.execute_command(fixed_command)
                
            # If the command executed successfully
            if success:
                # Verify that the fixed command achieved the intended purpose
                verification = self.verify_step_success(step, fixed_command, output, success)
                
                if verification['success']:
                    console.print("[bold green]Step fixed successfully![/bold green]")
                    return True
                else:
                    console.print(f"[bold yellow]Command executed but didn't achieve the goal: {verification['message']}[/bold yellow]")
                    
                    # If in manual mode, ask if the user wants to consider it a success anyway
                    if self.mode == "manual" and Confirm.ask("Consider this step successful anyway?", default=False):
                        return True
                    
                    return False
            else:
                console.print("[bold red]Fix attempt failed: Command execution failed.[/bold red]")
                return False
            
        except Exception as e:
            console.print(f"[bold red]Error generating or executing fix: {str(e)}[/bold red]")
            if self.debug:
                import traceback
                console.print(traceback.format_exc())
            return False
    
    def classify_command_type(self, command: str) -> str:
        """
        Classifies a command into categories: 
        - 'read_only': Commands that only read data (ls, find, grep, etc.)
        - 'information': Commands that display information (echo, printf, etc.)
        - 'write': Commands that write/modify files but don't delete (touch, mkdir, cp, etc.)
        - 'move': Commands that move files (mv, rename, etc.)
        - 'destructive': Commands that delete data (rm, rmdir, etc.)
        - 'combined': Commands with multiple operations joined by && or |
        - 'complex': Other commands that don't fit these categories
        
        Args:
            command: The command to classify
            
        Returns:
            str: The command classification
        """
        command = command.strip()
        
        # Read-only commands
        read_only_patterns = [
            r'^ls\s', r'^ls$', r'^find\s', r'^grep\s', r'^cat\s', 
            r'^head\s', r'^tail\s', r'^wc\s', r'^file\s',
            r'^awk\s', r'^sort(\s|$)', r'^uniq(\s|$)'  # Added common pipe commands
        ]
        
        # Information display commands
        info_patterns = [r'^echo\s', r'^printf\s', r'^date', r'^pwd', r'^whoami']
        
        # Write/modify commands
        write_patterns = [
            r'^touch\s', r'^mkdir\s', r'^cp\s', r'^rsync\s', 
            r'^chmod\s', r'^chown\s', r'^sed\s'
        ]
        
        # Move commands
        move_patterns = [r'^mv\s', r'^rename\s']
        
        # Destructive commands
        destructive_patterns = [r'^rm\s', r'^rmdir\s', r'^shred\s', r'\srm\s']
        
        # Check for command sequences with && or ;
        if '&&' in command or ';' in command:
            return 'combined'
        
        # Special handling for piped commands
        if '|' in command:
            # Split the command by pipes and analyze each part
            pipe_parts = [part.strip() for part in command.split('|')]
            
            # Check if all parts are read-only or information commands
            all_safe = True
            for part in pipe_parts:
                # Extract the base command (before any options)
                base_cmd = part.split()[0] if part.split() else ""
                
                # Check if this part matches any safe pattern
                part_is_safe = False
                
                # Check against read-only patterns
                for pattern in read_only_patterns:
                    if re.search(pattern.replace('^', ''), base_cmd):
                        part_is_safe = True
                        break
                        
                # Check against info patterns if not already matched
                if not part_is_safe:
                    for pattern in info_patterns:
                        if re.search(pattern.replace('^', ''), base_cmd):
                            part_is_safe = True
                            break
                
                # If any part is not safe, the whole command is not safe
                if not part_is_safe:
                    all_safe = False
                    break
            
            # If all parts are safe commands, classify as read_only
            if all_safe:
                return 'read_only'
            else:
                return 'combined'
        
        # Check individual command against patterns
        for pattern in read_only_patterns:
            if re.search(pattern, command):
                return 'read_only'
                
        for pattern in info_patterns:
            if re.search(pattern, command):
                return 'information'
                
        for pattern in write_patterns:
            if re.search(pattern, command):
                return 'write'
                
        for pattern in move_patterns:
            if re.search(pattern, command):
                return 'move'
                
        for pattern in destructive_patterns:
            if re.search(pattern, command):
                return 'destructive'
        
        return 'complex'
    
    def validate_command_for_phase(self, command: str, phase: str) -> Dict:
        """
        Validates if a command is appropriate for the current execution phase.
        
        Args:
            command: The command to validate
            phase: The current execution phase ('analysis', 'planning', 'execution')
            
        Returns:
            Dict: Validation result with keys:
                - valid (bool): Whether the command is valid for this phase
                - message (str): Explanation message
                - command_type (str): The classified command type
        """
        command_type = self.classify_command_type(command)
        
        if phase == 'analysis':
            # Analysis phase should only have read-only commands
            valid = command_type in ['read_only', 'information']
            message = (
                "Valid analysis command" if valid else 
                f"Invalid command for analysis phase - {command_type} operations should not be used in this phase"
            )
        
        elif phase == 'planning':
            # Planning phase should not have destructive or move commands
            valid = command_type not in ['destructive', 'move']
            message = (
                "Valid planning command" if valid else 
                f"Invalid command for planning phase - {command_type} operations should not be used in this phase"
            )
        
        elif phase == 'execution':
            # All commands are valid in execution phase
            valid = True
            message = "Valid execution command"
        
        else:
            valid = True
            message = "Unknown phase, command allowed by default"
        
        return {
            "valid": valid,
            "message": message,
            "command_type": command_type
        }


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
        "--max-logs",
        type=int,
        default=MAX_LOG_FILES,
        help="Maximum number of log files to keep"
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=str(LOGS_FOLDER),
        help="Directory where conversation logs are stored"
    )
    parser.add_argument(
        "--request",
        "-r",
        type=str,
        help="Run in one-shot mode with the provided request and exit after completion"
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Display the log after completion when running in one-shot mode"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Print configuration for debugging
    print(f"Using logs directory: {args.logs_dir}")
    print(f"Maximum log files to keep: {args.max_logs}")
    
    # Create a custom FileLogger with the user-specified settings
    custom_logger = FileLogger(
        logs_folder=Path(args.logs_dir),
        max_log_files=args.max_logs
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
    
    # Print successful logger setup confirmation
    print(f"AI Terminal initialized with custom logger at: {terminal.logger.logs_folder}")
    
    # Check if running in one-shot mode
    if args.request:
        console = Console()  # Create console instance for formatted output
        console.print(f"[bold green]AI Terminal One-Shot Mode[/bold green]")
        console.print(f"[bold]Request:[/bold] {args.request}")
        console.print("=" * 50)
        
        # Process the request with one-shot mode enabled for more verbose output
        terminal.process_request(args.request, one_shot_mode=True)
        
        console.print("\n" + "=" * 50)
        console.print("[bold green]Request completed. Output saved to log.[/bold green]")
        console.print(f"[dim]Log file: {terminal.logger.log_file}[/dim]")
        
        if args.show_log:
            console.print("\n[bold blue]One-Shot Mode Log:[/bold blue]")
            console.print(Panel(terminal.logger.get_conversation_history(), title=f"Log File: {terminal.logger.log_file}", expand=False))
    else:
        # Run in interactive mode
        terminal.run() 