import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


class EntropyAnalysisModelWrapper:
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
        pass


# Example usage
if __name__ == "__main__":
    model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    sampler = EntropyAnalysisModelWrapper(model_name, device='cuda')
    # Prepare input
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What number is bigger 9.11 or 9.9?"},
    ]
    tokenized_chat = sampler.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                           return_tensors="pt")

    # Generate text using the custom generate method
    output = sampler.generate(tokenized_chat.to("cuda"), max_length=500)

    # Decode the generated tokens
    generated_text = sampler.tokenizer.decode(output[0], skip_special_tokens=False)
    print(generated_text)

    output = sampler.model.generate(tokenized_chat.to("cuda"), max_length=500)

    generated_text = sampler.tokenizer.decode(output[0], skip_special_tokens=False)
    print(generated_text)
