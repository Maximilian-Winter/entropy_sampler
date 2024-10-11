import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import entropy

class EntropyAnalysisWrapper:
    def __init__(self, model_name, device="cuda"):
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True).to(device)
        self.model.eval()
        self.model_name = model_name
        self.device = device

        # Placeholders for entropy statistics
        self.logits_mean = None
        self.logits_std = None
        self.attn_mean = None
        self.attn_std = None

    def calculate_entropy(self, probs):
        # Ensure probabilities are non-zero
        probs = probs[probs > 0]
        return entropy(probs, base=2)

    def collect_calibration_data(self, input_output_pairs):
        # Lists to store entropy values
        logits_entropy_list = []
        attention_entropy_list = []

        # Process each input-output pair
        for input_text, output_text in input_output_pairs:
            # Combine input and output for the model
            full_text = input_text + output_text

            # Tokenize input and output
            inputs = self.tokenizer(input_text, return_tensors='pt').to("cuda")
            outputs = self.tokenizer(output_text, return_tensors='pt').to("cuda")

            # Get model outputs with attention
            with torch.no_grad():
                model_output = self.model(**inputs)

            # Get logits and attentions
            logits = model_output.logits  # [batch_size, seq_len, vocab_size]
            attentions = model_output.attentions  # List of tensors per layer

            # Calculate probabilities from logits
            probs = F.softmax(logits, dim=-1)

            # Entropy for the last token
            last_token_probs = probs[0, -1, :].cpu().numpy()
            logits_entropy = self.calculate_entropy(last_token_probs)
            logits_entropy_list.append(logits_entropy)

            # Attention entropy for the last layer and last token
            last_layer_attentions = attentions[-1]  # [batch_size, num_heads, seq_len, seq_len]
            avg_attn_weights = last_layer_attentions[0].mean(dim=0)  # [seq_len, seq_len]
            last_token_attn = avg_attn_weights[-1, :].cpu().numpy()
            attention_entropy = self.calculate_entropy(last_token_attn)
            attention_entropy_list.append(attention_entropy)

            print(f"Input: {input_text}")
            print(f"Output: {output_text}")
            print(f"Logits Entropy: {logits_entropy:.4f} bits")
            print(f"Attention Entropy: {attention_entropy:.4f} bits")
            print("-" * 50)

        # Calculate mean and standard deviation
        self.logits_mean = np.mean(logits_entropy_list)
        self.logits_std = np.std(logits_entropy_list)
        self.attn_mean = np.mean(attention_entropy_list)
        self.attn_std = np.std(attention_entropy_list)

        print(f"Logits Entropy - Mean: {self.logits_mean:.4f}, Std: {self.logits_std:.4f}")
        print(f"Attention Entropy - Mean: {self.attn_mean:.4f}, Std: {self.attn_std:.4f}")
        print("=" * 50)

    def categorize_state(self, logits_entropy, attention_entropy):
        # Define thresholds based on calibration data
        high_logits_entropy = self.logits_mean + self.logits_std
        low_logits_entropy = self.logits_mean - self.logits_std
        high_attn_entropy = self.attn_mean + self.attn_std
        low_attn_entropy = self.attn_mean - self.attn_std

        if logits_entropy > high_logits_entropy and attention_entropy > high_attn_entropy:
            return 'Uncertain'
        elif logits_entropy < low_logits_entropy and attention_entropy < low_attn_entropy:
            return 'Overconfident'
        else:
            return 'Confident'

    def analyze_model_state(self, input_text):
        # Tokenize input text
        inputs = self.tokenizer(input_text, return_tensors='pt')
        inputs["input_ids"] = inputs["input_ids"].to(self.device)

        # Get model outputs with attention
        with torch.no_grad():
            model_output = self.model(inputs["input_ids"])

        logits = model_output.logits
        attentions = model_output.attentions

        # Calculate probabilities from logits
        probs = F.softmax(logits, dim=-1)
        last_token_probs = probs[0, -1, :].cpu().numpy()
        logits_entropy = self.calculate_entropy(last_token_probs)

        # Attention entropy for the last layer and last token
        last_layer_attentions = attentions[-1]
        avg_attn_weights = last_layer_attentions[0].mean(dim=0)
        last_token_attn = avg_attn_weights[-1, :].cpu().numpy()
        attention_entropy = self.calculate_entropy(last_token_attn)

        state = self.categorize_state(logits_entropy, attention_entropy)

        print(f"Input: {input_text}")
        print(f"Logits Entropy: {logits_entropy:.4f} bits")
        print(f"Attention Entropy: {attention_entropy:.4f} bits")
        print(f"Model State: {state}")

    def generate_text(self, input_text, max_length=50, method='temperature', **kwargs):
        inputs = self.tokenizer(input_text, return_tensors='pt')

        # Prepare generation parameters
        generation_kwargs = {
            'input_ids': inputs['input_ids'].to(self.device),
            'max_length': max_length,
            'do_sample': True,  # Enable sampling
            "return_dict": True,
            'output_attentions': True,
            'output_hidden_states': True
        }

        if method == 'temperature':
            temperature = kwargs.get('temperature', 1.0)
            generation_kwargs['temperature'] = temperature
        elif method == 'top-k':
            k = kwargs.get('k', 50)
            generation_kwargs['top_k'] = k
        elif method == 'top-p':
            p = kwargs.get('p', 0.9)
            generation_kwargs['top_p'] = p
        elif method == 'min-p':
            # Implement custom min-p sampling
            min_p = kwargs.get('min_p', 0.02)
            generated_ids = inputs['input_ids'].to(self.device)
            for _ in range(max_length - generated_ids.shape[1]):
                with torch.no_grad():
                    outputs = self.model(input_ids=generated_ids)
                    logits = outputs.logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)

                    # Filter tokens with probability >= min_p
                    indices_to_keep = probs[0] >= min_p
                    if torch.sum(indices_to_keep) == 0:
                        # Fall back to argmax if no tokens meet the threshold
                        next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
                    else:
                        probs_filtered = probs[0] * indices_to_keep
                        probs_filtered = probs_filtered / probs_filtered.sum()
                        next_token_id = torch.multinomial(probs_filtered.unsqueeze(0), num_samples=1)

                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return generated_text
        else:
            raise ValueError("Invalid method. Choose from 'temperature', 'top-k', 'top-p', or 'min-p'.")

        # Generate text using built-in generation methods
        generated_outputs = self.model.generate(**generation_kwargs)
        generated_text = self.tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
        return generated_text



# Initialize the wrapper with a model name
wrapper = EntropyAnalysisWrapper('gpt2')

# Collect calibration data
input_output_pairs = [
    ("Once upon a time", " there was a brave knight."),
    ("The capital of France is", " Paris."),
    ("In quantum mechanics,", " particles can be in multiple states."),
    # Add more pairs as needed
]
wrapper.collect_calibration_data(input_output_pairs)

# Analyze model state at inference
wrapper.analyze_model_state("The quick brown fox")

# Generate text with different sampling methods
generated_text_temp = wrapper.generate_text("The meaning of life is", method='temperature', temperature=0.7)
print("Generated Text (Temperature Sampling):")
print(generated_text_temp)

generated_text_topk = wrapper.generate_text("The meaning of life is", method='top-k', k=50)
print("\nGenerated Text (Top-K Sampling):")
print(generated_text_topk)

generated_text_topp = wrapper.generate_text("The meaning of life is", method='top-p', p=0.9)
print("\nGenerated Text (Top-P Sampling):")
print(generated_text_topp)

generated_text_minp = wrapper.generate_text("The meaning of life is", method='min-p', min_p=0.01)
print("\nGenerated Text (Min-P Sampling):")
print(generated_text_minp)
