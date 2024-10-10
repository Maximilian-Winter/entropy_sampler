import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


class EntropyBasedSampler:
    def __init__(self, model_name, device='cuda'):
        # Load the pre-trained model and tokenizer
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode
        self.config = {
            'n_layers': self.model.config.num_hidden_layers,
            'n_heads': self.model.config.num_attention_heads,
            'head_dim': self.model.config.head_dim,
            'max_seq_len': 2048,
            'seed': 42,
            'verbose': True
        }

    def generate(self, input_ids, max_length=150):
        gen_tokens = input_ids.clone()

        # Initialize cache
        cache = DynamicCache()

        for _ in range(max_length - len(input_ids[0])):
            inputs = {'input_ids': gen_tokens,
                      'attention_mask': torch.ones_like(gen_tokens),
                      'return_dict': True,
                      'output_attentions': True,
                      'use_cache': True,
                      'past_key_values': cache}

            try:
                outputs = self.model(**inputs)
            except RuntimeError as e:
                print(f"Error during generation: {e}")
                break

            logits = outputs.logits
            attention_scores = outputs.attentions

            # Calculate Logit and Attention Score Shannon Entropy
            next_token = None # Sample with logit Shannon entropy, attention score Shannon entropy

            if self.config['verbose']:
                decoded_token = self.tokenizer.decode(next_token.squeeze())
                print(decoded_token, end="", flush=True)

            gen_tokens = torch.cat((gen_tokens, next_token), dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return gen_tokens[0].tolist()


# Example usage
if __name__ == "__main__":
    model_name = 'meta-llama/Llama-3.2-1B-Instruct'  # Replace with the desired model name
    sampler = EntropyBasedSampler(model_name, device='cuda')
    # Prepare input
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What number is bigger 9.11 or 9.9?"},
    ]
    tokenized_chat = sampler.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                           return_tensors="pt")

    # Generate text using the custom generate method
    output = sampler.generate(tokenized_chat, max_length=500)

    # Decode the generated tokens
    generated_text = sampler.tokenizer.decode(output[0], skip_special_tokens=False)
    print(generated_text)
