"""
Prompt Engineering with GPT-2
==============================
Demonstrates five prompt engineering techniques using the GPT-2 language
model from Hugging Face: zero-shot generation, custom sampling parameters,
few-shot learning, consistency analysis across related prompts, and
prompt chaining for narrative extension.

No external dataset required.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


MODEL_NAME  = "gpt2"
PAD_TOKEN   = 50256   # GPT-2 EOS token used as pad


# ── Model Initialisation ──────────────────────────────────────────────────────

def load_model(model_name: str = MODEL_NAME):
    """Load GPT-2 tokeniser and model."""
    tokeniser = GPT2Tokenizer.from_pretrained(model_name)
    model     = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    print(f"Loaded model: {model_name}")
    return tokeniser, model


def generate(prompt: str, tokeniser, model, max_length: int = 100,
             do_sample: bool = False, temperature: float = 1.0,
             num_return_sequences: int = 1) -> str:
    """Tokenise a prompt, generate text, and return the decoded output."""
    inputs = tokeniser.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            pad_token_id=PAD_TOKEN,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )
    return tokeniser.decode(outputs[0], skip_special_tokens=True)


# ── Technique 1: Zero-Shot Learning ──────────────────────────────────────────

def zero_shot_example(tokeniser, model):
    """
    Zero-shot: provide a task description as the prompt without
    any examples. The model generates a response based on its
    pre-trained knowledge alone.
    """
    prompt = "Write an advertising copy for a futuristic electric car with 100 words:"
    result = generate(prompt, tokeniser, model, max_length=100)
    print("\n[Zero-Shot] Advertising Copy:")
    print(result)
    return result


# ── Technique 2: Custom Sampling Parameters ───────────────────────────────────

def sampling_example(tokeniser, model):
    """
    Sampling with temperature > 1 produces more varied, creative output.
    Temperature < 1 makes the output more deterministic and focused.
    """
    prompt = "Write an advertising copy for a futuristic electric car with 100 words:"
    result = generate(prompt, tokeniser, model, max_length=100,
                      do_sample=True, temperature=1.5)
    print("\n[Sampling, temperature=1.5] Advertising Copy:")
    print(result)
    return result


# ── Technique 3: Few-Shot Learning ───────────────────────────────────────────

def few_shot_example(tokeniser, model):
    """
    Few-shot: supply labelled examples in the prompt to steer the model
    toward a specific output format and style.
    """
    few_shot_prompt = """
1. Title: The breakthrough in renewable energy
   Introduction: Scientists have made a significant breakthrough in solar cell technology...

2. Title: Discoveries in deep space exploration
   Introduction: The recent space mission has uncovered new insights into black holes...

3. Title: Advancements in Artificial Intelligence
   Introduction:"""

    inputs = tokeniser.encode(few_shot_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=200,
                                  num_return_sequences=1, pad_token_id=PAD_TOKEN)
    result = tokeniser.decode(outputs[0], skip_special_tokens=True)
    print("\n[Few-Shot] News Introduction:")
    print(result)
    return result


# ── Technique 4: Consistency Analysis ────────────────────────────────────────

def consistency_example(tokeniser, model):
    """
    Run several thematically related prompts and inspect how consistently
    the model handles a shared topic.
    """
    prompts = [
        "The first step to colonizing Mars is",
        "One challenge in Mars colonization is",
        "A potential solution for sustaining life on Mars is",
    ]
    print("\n[Consistency Analysis] Mars Colonization:")
    for prompt in prompts:
        result = generate(prompt, tokeniser, model, max_length=150)
        print(f"\nPrompt : {prompt}")
        print(f"Output : {result}")


# ── Technique 5: Prompt Chaining ─────────────────────────────────────────────

def chaining_example(tokeniser, model):
    """
    Prompt chaining: use part of a generated response as the input for
    the next generation step to produce an extended narrative.
    """
    initial_prompt = "In a distant future, humanity has discovered a way to"
    first_output   = generate(initial_prompt, tokeniser, model, max_length=100)

    # Extract the last complete sentence as the chained prompt
    sentences       = [s.strip() for s in first_output.split(".") if s.strip()]
    chained_prompt  = sentences[-2] + " which leads to" if len(sentences) >= 2 else first_output
    second_output   = generate(chained_prompt, tokeniser, model, max_length=100)

    print("\n[Prompt Chaining] Science Fiction Narrative:")
    print(f"Part 1: {first_output}")
    print(f"Part 2: {second_output}")
    return first_output, second_output


# ── Technique 6: Emotional Tone ──────────────────────────────────────────────

def emotional_tone_example(tokeniser, model):
    """
    Guide the model's emotional register by specifying the desired tone
    explicitly in the prompt.
    """
    prompt = "Write an optimistic view on the future of artificial intelligence in healthcare."
    result = generate(prompt, tokeniser, model, max_length=150)
    print("\n[Emotional Tone – Optimistic] AI in Healthcare:")
    print(result)
    return result


# ── Technique 7: Context-Aware Generation ────────────────────────────────────

def context_aware_example(tokeniser, model):
    """
    Provide rich context (audience, product features) so the model
    generates a more targeted product description.
    """
    prompt = """Product: Solar Powered Wireless Charger
Target Audience: Outdoor enthusiasts and environmentalists
Unique Feature: Can charge multiple devices simultaneously
Generate Product Description:"""

    result = generate(prompt, tokeniser, model, max_length=150)
    print("\n[Context-Aware] Product Description:")
    print(result)
    return result


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tokeniser, model = load_model()

    zero_shot_example(tokeniser, model)
    sampling_example(tokeniser, model)
    few_shot_example(tokeniser, model)
    consistency_example(tokeniser, model)
    chaining_example(tokeniser, model)
    emotional_tone_example(tokeniser, model)
    context_aware_example(tokeniser, model)
