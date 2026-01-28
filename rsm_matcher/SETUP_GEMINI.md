# Gemini API Setup Guide

## Get Your Free API Key (2 minutes)

1. **Go to Google AI Studio**: https://aistudio.google.com/app/apikey
2. **Sign in** with your Google account
3. **Click "Get API Key"** or "Create API Key"
4. **Copy the key** (starts with `AIza...`)

## Set the API Key

### Windows (Command Prompt):
```cmd
set GEMINI_API_KEY=your-key-here
```

### Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="your-key-here"
```

### Linux/Mac:
```bash
export GEMINI_API_KEY='your-key-here'
```

### Make it Permanent (Optional):

**Windows:**
1. Search "Environment Variables" in Start Menu
2. Click "Edit the system environment variables"
3. Click "Environment Variables" button
4. Under "User variables", click "New"
5. Variable name: `GEMINI_API_KEY`
6. Variable value: Your API key
7. Click OK

**Linux/Mac:**
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export GEMINI_API_KEY='your-key-here'
```

## Install the Package

```bash
pip install google-generativeai
```

## Usage Limits (Free Tier)
- 15 requests per minute
- 1 million tokens per day
- Perfect for this demo!

## Without API Key
The program will still work using template-based explanations.
LLM features are optional but recommended for better quality.
