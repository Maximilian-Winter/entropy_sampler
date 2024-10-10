import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


class EntropyBasedSampler:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 entropy_min=1.0, entropy_max=5.0, temp_min=0.7, temp_max=1.5, beta=0.9):
        """
        Wrapper for a language model to perform entropy-based sampling.

        Args:
            model (PreTrainedModel): The language model.
            tokenizer (PreTrainedTokenizer): The tokenizer.
            entropy_min (float): Minimum expected entropy.
            entropy_max (float): Maximum expected entropy.
            temp_min (float): Minimum temperature to use.
            temp_max (float): Maximum temperature to use.
            beta (float): Smoothing factor for moving average (0 < beta < 1).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.beta = beta
        self.prev_entropy = None  # For moving average

    def generate(self, input_ids, max_length=50, pad_token_id=None, eos_token_id=None):
        """
        Generates text using entropy-based sampling.

        Args:
            input_ids (torch.LongTensor): Input prompt token IDs.
            max_length (int): Maximum length of the generated sequence.
            pad_token_id (int, optional): Padding token ID.
            eos_token_id (int, optional): End-of-sequence token ID.

        Returns:
            torch.LongTensor: Generated token IDs.
        """
        generated = input_ids
        batch_size = input_ids.size(0)

        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        # Initialize past_key_values for efficient decoding
        past = None

        for _ in range(max_length):
            outputs = self.model(input_ids=generated[:, -1:], past_key_values=past, return_dict=True)

            # Update past_key_values
            past = outputs.past_key_values

            logits = outputs.logits[:, -1, :]  # Get logits of the last token

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, vocab_size)

            # Calculate entropy for each item in the batch
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # Shape: (batch_size,)

            # Initialize or update the moving average of entropy
            if self.prev_entropy is None:
                self.prev_entropy = entropy
            else:
                self.prev_entropy = self.beta * self.prev_entropy + (1 - self.beta) * entropy

            # Normalize entropy to [0, 1]
            H_norm = (self.prev_entropy - self.entropy_min) / (self.entropy_max - self.entropy_min)
            H_norm = torch.clamp(H_norm, 0.0, 1.0)

            # Adjust temperature inversely to normalized entropy
            temperature = self.temp_max - H_norm * (self.temp_max - self.temp_min)  # Shape: (batch_size,)
            temperature = temperature.unsqueeze(-1)  # Shape: (batch_size, 1)

            # Adjust logits using the temperature
            adjusted_logits = logits / temperature

            # Sample the next token
            next_token = torch.multinomial(F.softmax(adjusted_logits, dim=-1), num_samples=1)  # Shape: (batch_size, 1)
            # Append to generated tokens
            generated = torch.cat((generated, next_token), dim=1)
            print(self.tokenizer.decode(next_token.item()), end="")

            # Check for EOS token
            if (next_token == eos_token_id).all():
                break

        return generated


from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to("cuda")

entropy_sampler_model = EntropyBasedSampler(model, tokenizer, entropy_min=0.1, entropy_max=2.5, temp_min=0.25, temp_max=1.0, beta=0.9)

# Prepare input
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What number is bigger 9.11 or 9.9?"},
]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

output_ids = entropy_sampler_model.generate(
    tokenized_chat.to("cuda"),
    tokenizer.eos_token_id,
    tokenizer.eos_token_id
)

# Decode and print the result
print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
