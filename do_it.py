import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math


class EntropyBasedSampler:
    def __init__(self, model_name, device='cuda'):
        # Load the pre-trained model and tokenizer
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode

        # Define parameter bounds
        self.temp_min = 0.25
        self.temp_max = 1.25
        self.top_p_min = 0.75
        self.top_p_max = 1.0
        self.top_k_min = 40
        self.top_k_max = 100

        # Initialize previous entropy for smoothing
        self.prev_entropy = None
        self.alpha = 0.25  # Smoothing factor

    def compute_entropy_from_logits(self, logits):
        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=-1)
        # Compute entropy
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)
        return entropy

    def compute_entropy_from_attention(self, attentions):
        # attentions: list of attention tensors from each layer
        entropy_list = []
        for layer_attention in attentions:
            # layer_attention shape: (batch_size, num_heads, seq_len, seq_len)
            # Average over heads
            attention_probs = torch.mean(layer_attention, dim=1)  # Shape: (batch_size, seq_len, seq_len)
            # Compute entropy for each position
            entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-10),
                                 dim=-1)  # Shape: (batch_size, seq_len)
            # Average over positions
            entropy = torch.mean(entropy, dim=-1)  # Shape: (batch_size,)
            entropy_list.append(entropy)
        # Average over layers
        avg_entropy = torch.mean(torch.stack(entropy_list), dim=0)
        return avg_entropy

    def adjust_parameters(self, entropy):
        # Normalize entropy to [0, 1]
        entropy_norm = (entropy - self.entropy_min) / (self.entropy_max - self.entropy_min)
        entropy_norm = torch.clamp(entropy_norm, 0, 1)

        # Linear scaling for temperature
        temperature = self.temp_min + (self.temp_max - self.temp_min) * entropy_norm

        # Linear scaling for top_p
        top_p = self.top_p_min + (self.top_p_max - self.top_p_min) * entropy_norm

        # Linear scaling for top_k
        top_k = int(self.top_k_min + (self.top_k_max - self.top_k_min) * entropy_norm)
        return temperature.item(), top_p.item(), top_k

    def smooth_entropy(self, entropy):
        if self.prev_entropy is None:
            smoothed_entropy = entropy
        else:
            smoothed_entropy = self.alpha * entropy + (1 - self.alpha) * self.prev_entropy
        self.prev_entropy = smoothed_entropy
        return smoothed_entropy

    def generate(self, input_ids, max_length=50):
        generated = input_ids.to(self.device)
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare model inputs
                inputs = {'input_ids': generated, 'return_dict': True, 'output_attentions': True}
                outputs = self.model(**inputs)

                # Get next token logits
                logits = outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)

                # Compute entropy from logits
                logits_entropy = self.compute_entropy_from_logits(logits)  # Shape: (batch_size,)

                # Compute entropy from attentions
                attentions = outputs.attentions  # List of tensors
                attention_entropy = self.compute_entropy_from_attention(attentions)  # Shape: (batch_size,)

                # Average the two entropies
                entropy = (logits_entropy + attention_entropy) / 2.0  # Shape: (batch_size,)

                # Smooth the entropy
                entropy = self.smooth_entropy(entropy)

                # Set entropy min and max for normalization (empirically determined or set)
                self.entropy_min = 1.0  # Adjust these values as needed
                self.entropy_max = 3.5

                # Adjust sampling parameters based on entropy
                temperature, top_p, top_k = self.adjust_parameters(entropy)

                # Adjust logits with temperature
                logits = logits / temperature

                # Apply top_k sampling
                if top_k > 0:
                    top_k = max(1, top_k)
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')

                # Apply nucleus (top_p) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[0, indices_to_remove] = -float('Inf')

                # Sample from the adjusted distribution
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                # Decode the token and print
                decoded_token = self.tokenizer.decode(next_token.squeeze())
                print(decoded_token, end="", flush=True)
                # Append the sampled token to the input
                generated = torch.cat((generated, next_token), dim=1)

                # If end-of-sentence token is generated, stop
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return generated


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
    # print(generated_text)
