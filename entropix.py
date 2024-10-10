from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from EntropixHugginface import EntropixHuggingFace

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")


entropy_sampler = EntropixHuggingFace("meta-llama/Llama-3.2-3B-Instruct")

# Prepare input
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What number is bigger 9.11 or 9.9?"},
]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
attention_mask = tokenized_chat.ne(tokenizer.eos_token_id).long()

# Set up generation config
generation_config = GenerationConfig(
    max_new_tokens=250,
    do_sample=True,
    temperature=0.45,
    top_p=0.95,
    num_return_sequences=1,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
)

# Generate output
output_ids = entropy_sampler.generate(
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
    max_length=200
)

# Decode and print the result
print(tokenizer.decode(output_ids[0], skip_special_tokens=False))