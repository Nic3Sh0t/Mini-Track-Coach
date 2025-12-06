# Configuration Setup Guide

## API Keys Configuration

To use AI coaching advice features, you need to configure API keys in the configuration file.

### Step 1: Create Configuration File

Copy the example configuration file:

```bash
cp llamacloud_config.json.example llamacloud_config.json
```

### Step 2: Fill in Your API Keys

Edit `llamacloud_config.json` and replace the placeholder values:

```json
{
  "name": "your_index_name",
  "project_name": "your_project_name",
  "organization_id": "your_organization_id",
  "api_key": "your_llamacloud_api_key_here",
  "llm": {
    "provider": "openrouter",
    "api_key": "your_openrouter_api_key_here",
    "model": "deepseek/deepseek-v3.2",
    "temperature": 0.7,
    "max_tokens": 2048,
    "context_window": 4096
  }
}
```

### Required Fields

- **name**: Your LlamaCloud index name
- **project_name**: Your LlamaCloud project name
- **organization_id**: Your LlamaCloud organization ID
- **api_key**: Your LlamaCloud API key
- **llm.api_key**: Your OpenRouter API key (or other LLM provider API key)

### Optional Fields

- **llm.model**: LLM model to use (default: "deepseek/deepseek-v3.2")
- **llm.temperature**: Sampling temperature (default: 0.7)
- **llm.max_tokens**: Maximum tokens in response (default: 2048)
- **llm.context_window**: Context window size (default: 4096)

### Security Notes

⚠️ **IMPORTANT**: 
- The `llamacloud_config.json` file is already in `.gitignore` and will **NOT** be committed to the repository
- **Never commit API keys** to version control
- Keep your API keys secure and do not share them publicly
- If you accidentally commit an API key, rotate it immediately

### Usage

Once configured, you can use AI coaching advice features:

```bash
python -m trackcoach.pipeline \
  --csv data.csv \
  --outdir output \
  --generate-advice \
  --advice-config llamacloud_config.json
```

Or in the Web UI, enable "Generate AI Advice" and specify the config file path.

### Troubleshooting

If you encounter errors:
1. Verify that `llamacloud_config.json` exists and is properly formatted JSON
2. Check that all required fields are filled in (not placeholder values)
3. Ensure your API keys are valid and have the necessary permissions
4. Check the logs for detailed error messages
