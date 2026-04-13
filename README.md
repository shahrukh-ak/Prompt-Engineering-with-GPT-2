# Prompt Engineering with GPT-2

Demonstrates seven prompt engineering techniques using the GPT-2 language model. Each function is a self-contained example of a distinct prompting strategy, with printed outputs for direct comparison.

## Techniques Covered

| Technique | Description |
|-----------|-------------|
| Zero-shot | Task described in the prompt with no examples |
| Custom sampling | Temperature adjustment for creative vs focused output |
| Few-shot | Examples embedded in the prompt to guide format and style |
| Consistency analysis | Multiple related prompts to assess thematic coherence |
| Prompt chaining | Output from one generation used as input for the next |
| Emotional tone | Tone specified explicitly in the prompt |
| Context-aware | Structured product/audience context injected into the prompt |

## Model

GPT-2 (base) from OpenAI, loaded via Hugging Face `transformers`. The base variant is used throughout. Larger variants (`gpt2-medium`, `gpt2-large`) can be swapped in by changing the `MODEL_NAME` constant.

## Project Structure

```
15_gpt2_prompt_engineering/
├── gpt2_prompt_engineering.py  # All techniques in a single script
├── requirements.txt
└── README.md
```

## Requirements

```
transformers
torch
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

No dataset is required. Run:

```bash
python gpt2_prompt_engineering.py
```

All outputs are printed to stdout. Generation parameters (max_length, temperature) can be adjusted per function call.

## Notes

GPT-2 is a relatively small language model and its outputs reflect the capabilities of the base architecture. The purpose of this project is to demonstrate prompting techniques, not to benchmark state-of-the-art generation quality. The same techniques apply directly to larger models.
