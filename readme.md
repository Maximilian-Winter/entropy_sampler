# Entropy-based Sampler for Language Models

This repository contains an implementation of an entropy-based sampler for language models, inspired by the entropix project. It is implemented as a logits processor for Hugginface and the PyTorch Library. The sampler uses entropy and varentropy metrics to dynamically adjust sampling parameters during text generation, potentially leading to more coherent and context-aware outputs.

I used Claude Sonnet 3.5 to help me with the code and the math behind it.

Currently should work with llama 3.2 models. Can be adopted for other models realtivly easy, I think, by adopting the attention map hooks.

## Key Concepts

1. **Entropy and Varentropy**: Measures of uncertainty in the model's predictions.
2. **Attention Metrics**: Utilizes attention scores to calculate additional metrics like attention entropy and agreement.
3. **Adaptive Sampling**: Adjusts sampling parameters (temperature, top-k, top-p, min-p) based on the current state of the generation.
4. **Four Sampling Modes**:
- Low Entropy, Low Varentropy: "flowing with unspoken intent"
- High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
- Low Entropy, High Varentropy: "exploring forks in the path"
- High Entropy, High Varentropy: "resampling in the mist"

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/entropy-sampler.git
cd entropy-sampler
```


## Usage

Here's a basic example of how to use the entropy-based sampler:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from entropy_sampler import HookBasedEntropySampler

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Initialize the sampler
pause_token_id = tokenizer.convert_tokens_to_ids("...")
clarifying_question_token_id = tokenizer.convert_tokens_to_ids("?")
entropy_sampler = HookBasedEntropySampler(model, tokenizer, pause_token_id, clarifying_question_token_id)

# Prepare input
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What number is bigger 9.11 or 9.9?"},
]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
attention_mask = tokenized_chat.ne(tokenizer.eos_token_id).long()

# Set up generation config
generation_config = GenerationConfig(
    max_new_tokens=500,
    do_sample=True,
    temperature=0.45,
    top_p=0.95,
    num_return_sequences=1,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
)

# Generate output
output_ids = model.generate(
    tokenized_chat,
    attention_mask=attention_mask,
    generation_config=generation_config,
    logits_processor=[entropy_sampler]
)

# Decode and print the result
print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgements

This project is inspired by the entropix implementation.
