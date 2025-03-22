#!/usr/bin/env python3
"""
Setup configuration for AI Terminal Agent.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read the long description from README.md if it exists
long_description = ''
readme_path = Path('README.md')
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="ai-terminal-agent",
    version="0.1.0",
    description="AI-powered terminal agent that converts natural language to shell commands",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Terminal Team",
    author_email="your-email@example.com",
    url="https://github.com/yourusername/ai_terminal",
    packages=find_packages(),
    py_modules=["ai_terminal"],  # Include the main script as a module
    install_requires=requirements,
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "agent=agent_cli.cli:main",
            "agent-completion=agent_cli.completion:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Utilities",
    ],
) 