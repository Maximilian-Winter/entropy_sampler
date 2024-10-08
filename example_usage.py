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