#!/usr/bin/env python3
"""
Simple tests for the AI Terminal CLI.
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestCLI(unittest.TestCase):
    """Tests for the CLI module."""
    
    def test_cli_module_import(self):
        """Test that the CLI module can be imported."""
        try:
            from agent_cli.cli import main
            self.assertTrue(callable(main))
        except ImportError as e:
            self.fail(f"Failed to import CLI module: {e}")
    
    def test_ai_terminal_import(self):
        """Test that the AI Terminal module can be imported."""
        try:
            from ai_terminal import AITerminal, FileLogger
            self.assertTrue(callable(AITerminal))
            self.assertTrue(callable(FileLogger))
        except ImportError as e:
            self.fail(f"Failed to import AI Terminal: {e}")

if __name__ == "__main__":
    unittest.main() 