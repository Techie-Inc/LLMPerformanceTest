# LLM Performance Testing Tool

A Python tool for testing the performance and concurrency capabilities of local Large Language Models (LLMs) running on LM Studio.

## Features

- Test multiple concurrent connection levels
- Random prompt selection from a prompt file
- Detailed performance metrics including:
  - Average response time
  - Median response time
  - 95th percentile response time
  - Success/failure rates
  - Tokens per second
- Beautiful console output using Rich
- JSON results export

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd llm-performance-tester
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env-template` to `.env` and configure your settings:

```bash
cp .env-template .env
```

Edit `.env` with your preferred settings:
- `LLM_PROVIDER`: Choose between 'local' or 'openai'
- `LOCAL_LLM_URL`: URL for your local LM Studio instance
- `OPENAI_API_KEY`: Your OpenAI API key (if using OpenAI)
- `OPENAI_MODEL`: OpenAI model to use (if using OpenAI)

## Usage with Different Providers

### Local LLM (LM Studio)
```bash
python llm_performance_test.py --provider local
```

### OpenAI
```bash
python llm_performance_test.py --provider openai --model gpt-3.5-turbo
```

## Usage

### Basic Usage

```bash
python llm_performance_test.py
```

This will run with default settings:
- Testing up to 10 concurrent connections
- Using localhost:1234 as the LLM server
- Using test_prompts.txt for prompts
- Testing each concurrency level once

### Advanced Usage

```bash
python llm_performance_test.py \
  --url http://localhost:1234 \
  --max-concurrent 100 \
  --concurrent-step 20 \
  --requests-per-level 3 \
  --prompts-file custom_prompts.txt
```

### Command Line Arguments

- `--url`: LM Studio server URL (default: http://localhost:1234)
- `--max-concurrent`: Maximum number of concurrent connections to test (default: 10)
- `--concurrent-step`: Step size between concurrent connection tests (default: 1)
- `--requests-per-level`: Number of times to repeat each concurrency level test (default: 1)
- `--prompts-file`: File containing test prompts (default: test_prompts.txt)

### Example

Testing from 10 to 100 concurrent connections in steps of 10:
```bash
python llm_performance_test.py --max-concurrent 100 --concurrent-step 10
```

This will:
1. Test with up to 100 concurrent connections
2. Display results in a table
3. Show overall statistics
4. Save detailed results to llm_performance_results.json

## Prompt File Format

Create a text file (default: test_prompts.txt) with one prompt per line. The tool will randomly select prompts from this file for testing. Example:

```text
Tell me a joke about programming.
What is the meaning of life?
Explain quantum computing in simple terms.
Write a haiku about artificial intelligence.
```

## Output

The tool provides:
1. A real-time progress display
2. A formatted table showing results for each concurrency level
3. Overall statistics including:
   - Total test duration
   - Total requests made
   - Average time per request
4. A JSON file (llm_performance_results.json) with detailed results

## Requirements

- Python 3.7+
- aiohttp
- rich

## Notes

- Ensure your LM Studio instance is running before starting the tests
- The tool uses the chat completions API endpoint (/v1/chat/completions)
- Response times include network latency
- Token counts are estimated based on word count