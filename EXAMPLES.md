# AI Terminal Usage Examples

This document provides examples of how to use the AI Terminal for various common tasks on macOS.

## Basic File System Operations

### Finding Files

```
Find all images in my Downloads folder
```

This will execute a command like:
```bash
find ~/Downloads -type f -name "*.jpg" -o -name "*.png" -o -name "*.gif" -o -name "*.jpeg"
```

### Managing Disk Space

```
Show the largest files in my home directory
```

This will execute a command like:
```bash
find ~/ -type f -size +100M -exec du -h {} \; | sort -hr | head -10
```

## Development Tasks

### Setting Up Development Environments

```
Set up a Python virtual environment for a Flask project
```

This will create a series of commands like:
```bash
mkdir -p ~/flask_project
cd ~/flask_project
python3 -m venv venv
source venv/bin/activate
pip install flask
```

### Git Operations

```
Create a new git repository, add all files, and make an initial commit
```

This will execute:
```bash
git init
git add .
git commit -m "Initial commit"
```

## System Maintenance

### Checking System Status

```
Show me system information and resource usage
```

This will display:
```bash
system_profiler SPHardwareDataType && top -l 1 | head -10
```

### Finding and Killing Processes

```
Find what's using port 3000 and kill it
```

This will execute:
```bash
lsof -i :3000 && kill -9 $(lsof -t -i:3000)
```

## Application Installation

```
Install the latest version of Node.js using Homebrew
```

This will perform:
```bash
brew update && brew install node
```

## Network Operations

```
Show available WiFi networks
```

This will execute:
```bash
/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -s
```

## Complex Tasks (Agent Mode)

For complex tasks, the AI Terminal will automatically switch to Agent Mode, which breaks down the task into multiple steps:

```
Clone a GitHub repository, install its dependencies, and run the tests
```

This will create a plan like:
1. Ask for the repository URL
2. Clone the repository
3. Navigate to the repository directory
4. Detect the project type (Node.js, Python, etc.)
5. Install dependencies based on project type
6. Run appropriate test commands

## Security Operations

```
Generate a strong SSH key
```

This will execute:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

## Tips for Effective Use

1. **Be as specific as possible** in your requests.
2. Use **Manual Mode** when trying new operations to verify commands.
3. For complex tasks, use **Agent Mode** which automatically breaks down multi-step processes.
4. If a command fails, the AI Terminal will try to **diagnose and fix** the issue.
5. Use **natural language** - no need to learn specific command syntax. 