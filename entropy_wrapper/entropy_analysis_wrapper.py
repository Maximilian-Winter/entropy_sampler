import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns


class EntropyAnalysisWrapper:
    def __init__(self, model_name, device="cuda"):
        """
        Initialize the EntropyAnalysisWrapper with a pre-trained model and tokenizer.
        """
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            return_dict_in_generate=True,
            output_attentions=True,
            output_hidden_states=True
        ).to(device)
        self.model.eval()
        self.model_name = model_name
        self.device = device

        # Placeholders for entropy statistics
        self.logits_mean = None
        self.logits_std = None
        self.attn_mean = None
        self.attn_std = None

    def calculate_entropy(self, probs):
        """
        Calculate Shannon entropy of a probability distribution.
        """
        # Ensure probabilities are non-zero
        probs = probs[probs > 0]
        return entropy(probs, base=2)

    def calculate_sequence_entropy(self, probs):
        """
        Calculate entropy for each token in a sequence.
        """
        # probs: [seq_len, vocab_size]
        entropies = []
        for token_probs in probs:
            token_probs = token_probs[token_probs > 0]
            entropies.append(entropy(token_probs.cpu().numpy(), base=2))
        return entropies

    def collect_calibration_data(self, input_output_pairs):
        """
        Collect calibration data to compute mean and std of logits and attention entropies.
        """
        # Lists to store entropy values
        logits_entropy_list = []
        attention_entropy_list = []

        # Process each input-output pair
        for input_text, output_text in input_output_pairs:
            # Combine input and output for the model
            full_text = input_text + output_text

            # Tokenize full text
            inputs = self.tokenizer(full_text, return_tensors='pt').to(self.device)

            # Get model outputs with attention
            with torch.no_grad():
                model_output = self.model(**inputs)

            # Get logits and attentions
            logits = model_output.logits  # [batch_size, seq_len, vocab_size]
            attentions = model_output.attentions  # List of tensors per layer

            # Calculate probabilities from logits
            probs = F.softmax(logits, dim=-1)

            # Entropy for all tokens
            sequence_probs = probs[0]  # [seq_len, vocab_size]
            logits_entropies = self.calculate_sequence_entropy(sequence_probs)
            logits_entropy_list.extend(logits_entropies)

            # Attention entropy for each token
            last_layer_attentions = attentions[-1][0].mean(dim=0)  # [seq_len, seq_len]
            attention_entropies = []
            for idx in range(last_layer_attentions.size(0)):
                attn_weights = last_layer_attentions[idx].cpu().numpy()
                # Normalize attention weights
                attn_weights = attn_weights / attn_weights.sum()
                # Ensure probabilities are non-zero
                attn_weights = attn_weights[attn_weights > 0]
                entropy_val = self.calculate_entropy(attn_weights)
                attention_entropies.append(entropy_val)
            attention_entropy_list.extend(attention_entropies)

            print(f"Input: {input_text}")
            print(f"Output: {output_text}")
            print(f"Logits Entropies: {logits_entropies}")
            print(f"Attention Entropies: {attention_entropies}")
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
        """
        Categorize model state based on entropy thresholds.
        """
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
        """
        Analyze the model's state given an input text.
        """
        # Tokenize input text
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)

        # Get model outputs with attention and hidden states
        with torch.no_grad():
            model_output = self.model(**inputs)

        logits = model_output.logits
        attentions = model_output.attentions
        hidden_states = model_output.hidden_states

        # Calculate probabilities from logits
        probs = F.softmax(logits, dim=-1)
        sequence_probs = probs[0]
        logits_entropies = self.calculate_sequence_entropy(sequence_probs)

        # Attention entropy for each token
        last_layer_attentions = attentions[-1][0].mean(dim=0)  # [seq_len, seq_len]
        attention_entropies = []
        for idx in range(last_layer_attentions.size(0)):
            attn_weights = last_layer_attentions[idx].cpu().numpy()
            # Normalize attention weights
            attn_weights = attn_weights / attn_weights.sum()
            attn_weights = attn_weights[attn_weights > 0]
            entropy_val = self.calculate_entropy(attn_weights)
            attention_entropies.append(entropy_val)

        # Analyze hidden states
        hidden_state_stats = self.analyze_hidden_states(hidden_states)

        # Visualize attention
        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        self.visualize_attention(attentions, input_tokens)

        # Output results
        print(f"Input: {input_text}")
        print(f"Logits Entropies: {logits_entropies}")
        print(f"Attention Entropies: {attention_entropies}")
        for stat in hidden_state_stats:
            print(f"Layer {stat['layer']} - Mean Activation: {stat['mean_activation']:.4f}, "
                  f"Std Activation: {stat['std_activation']:.4f}")

        # Optional: Analyze attention heads
        head_entropies = self.analyze_attention_heads(attentions)
        print(f"Head Entropies: {head_entropies}")

        # Optional: Compute model uncertainty
        mean_probs, var_probs = self.mc_dropout(inputs)
        uncertainty = var_probs[0, -1, :].mean().item()
        print(f"Model Uncertainty (Variance): {uncertainty}")

        # Optional: Layer-wise activation stats
        layer_stats = self.layer_wise_activation_stats(hidden_states)
        for ls in layer_stats:
            print(f"Layer {ls['layer']} - Norm Activation: {ls['norm_activation']:.4f}")

        # Optionally categorize state for last token
        logits_entropy_last = logits_entropies[-1]
        attention_entropy_last = attention_entropies[-1]
        state = self.categorize_state(logits_entropy_last, attention_entropy_last)
        print(f"Model State: {state}")

    def analyze_hidden_states(self, hidden_states):
        """
        Analyze hidden states to compute mean and std of activations per layer.
        """
        layer_stats = []
        for idx, layer_hidden in enumerate(hidden_states):
            # layer_hidden: [batch_size, seq_len, hidden_size]
            mean_activation = layer_hidden.mean().item()
            std_activation = layer_hidden.std().item()
            layer_stats.append({
                'layer': idx,
                'mean_activation': mean_activation,
                'std_activation': std_activation
            })
        return layer_stats

    def visualize_attention(self, attentions, input_tokens):
        """
        Visualize the attention weights.
        """
        # attentions: list of tensors (per layer)
        # input_tokens: list of tokens
        last_layer_attn = attentions[-1][0]  # [num_heads, seq_len, seq_len]
        avg_attn = last_layer_attn.mean(dim=0)  # [seq_len, seq_len]

        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_attn.cpu().numpy(), xticklabels=input_tokens, yticklabels=input_tokens)
        plt.title('Attention Heatmap')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        plt.show()

    def analyze_attention_heads(self, attentions):
        """
        Analyze attention entropies per head.
        """
        head_entropies = []
        last_layer_attn = attentions[-1][0]  # [num_heads, seq_len, seq_len]
        num_heads = last_layer_attn.size(0)
        for head in range(num_heads):
            head_attn = last_layer_attn[head]
            attn_probs = head_attn[-1, :].cpu().numpy()
            attn_probs = attn_probs / attn_probs.sum()
            attn_probs = attn_probs[attn_probs > 0]
            head_entropy = self.calculate_entropy(attn_probs)
            head_entropies.append(head_entropy)
        return head_entropies

    def mc_dropout(self, inputs, n_samples=10):
        """
        Perform Monte Carlo Dropout to estimate model uncertainty.
        """
        # Set dropout layers to train mode
        self.model.train()
        dropout_outputs = []
        for _ in range(n_samples):
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            dropout_outputs.append(probs)
        # Set model back to eval mode
        self.model.eval()
        # Compute mean and variance
        probs_stack = torch.stack(dropout_outputs, dim=0)
        mean_probs = probs_stack.mean(dim=0)
        var_probs = probs_stack.var(dim=0)
        return mean_probs, var_probs

    def assess_calibration(self, true_labels, predicted_probs):
        """
        Assess model calibration using reliability diagrams.
        """
        confidences = [probs[label] for probs, label in zip(predicted_probs, true_labels)]
        # Create bins
        bins = np.linspace(0.0, 1.0, 11)  # 10 bins between 0 and 1
        bin_indices = np.digitize(confidences, bins) - 1  # Adjust indices to start from 0
        accuracy_per_bin = []
        for b in range(len(bins) - 1):
            indices = [i for i, x in enumerate(bin_indices) if x == b]
            if indices:
                bin_accuracy = np.mean([true_labels[i] == np.argmax(predicted_probs[i]) for i in indices])
                accuracy_per_bin.append(bin_accuracy)
            else:
                accuracy_per_bin.append(0)
        # Plot reliability diagram
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.figure()
        plt.plot(bin_centers, accuracy_per_bin, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Model Calibration')
        plt.legend()
        plt.show()

    def gradient_importance(self, input_text):
        """
        Compute gradient-based importance scores for input tokens.
        """
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
        embeddings.requires_grad_(True)
        outputs = self.model(inputs_embeds=embeddings)
        loss = outputs.logits[:, -1, :].mean()
        loss.backward()
        gradients = embeddings.grad
        # Normalize and visualize gradients
        gradient_importance = gradients.abs().sum(dim=-1).squeeze().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        plt.bar(range(len(tokens)), gradient_importance)
        plt.xticks(range(len(tokens)), tokens, rotation='vertical')
        plt.title('Gradient Importance')
        plt.show()

    def calculate_perplexity(self, input_text):
        """
        Calculate the perplexity of the input text.
        """
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
        perplexity = torch.exp(loss).item()
        print(f"Perplexity: {perplexity}")

    def layer_wise_activation_stats(self, hidden_states):
        """
        Compute layer-wise activation statistics.
        """
        layer_stats = []
        for idx, layer_hidden in enumerate(hidden_states):
            # Assume layer_hidden: [batch_size, seq_len, hidden_size]
            mean_activation = layer_hidden.mean().item()
            std_activation = layer_hidden.std().item()
            norm_activation = layer_hidden.norm().item()
            layer_stats.append({
                'layer': idx,
                'mean_activation': mean_activation,
                'std_activation': std_activation,
                'norm_activation': norm_activation
            })
        return layer_stats

    def generate_text(self, input_text, max_length=50, method='temperature', **kwargs):
        """
        Generate text using different sampling methods.
        """
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        attention_mask = inputs['input_ids'].ne(self.tokenizer.eos_token_id).long()
        # Prepare generation parameters
        generation_kwargs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': attention_mask,
            'max_length': max_length,
            'do_sample': True,  # Enable sampling
            'return_dict_in_generate': True,
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
            min_p = kwargs.get('min_p', 0.02)

            class MinPLogitsProcessor(LogitsProcessor):
                def __init__(self, min_p):
                    self.min_p = min_p

                def __call__(self, input_ids, scores):
                    probs = F.softmax(scores, dim=-1)
                    mask = probs >= self.min_p
                    if mask.sum() == 0:
                        # If no tokens meet the threshold, do not filter
                        return scores
                    else:
                        scores = scores.masked_fill(~mask, -float('inf'))
                        return scores

            logits_processor = MinPLogitsProcessor(min_p)
            generation_kwargs['logits_processor'] = [logits_processor]
        else:
            raise ValueError("Invalid method. Choose from 'temperature', 'top-k', 'top-p', or 'min-p'.")

        # Generate text using built-in generation methods
        generated_outputs = self.model.generate(**generation_kwargs)
        generated_text = self.tokenizer.decode(generated_outputs.sequences[0], skip_special_tokens=True)
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
