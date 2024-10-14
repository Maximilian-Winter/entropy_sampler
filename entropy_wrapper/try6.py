import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigurableAnalysis:
    """Configuration class for enabling or disabling specific analyses."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled


class EntropyAnalysisConfig:
    """Configuration for the entire entropy analysis."""

    def __init__(self):
        self.logits_entropy = ConfigurableAnalysis()
        self.attention_entropy = ConfigurableAnalysis()
        self.mc_dropout = ConfigurableAnalysis()
        self.perplexity = ConfigurableAnalysis()


class BaseEntropyAnalysisWrapper(ABC):
    """Abstract base class for entropy analysis wrappers."""

    def __init__(self, model_name: str, device: str = "cuda", config: Optional[EntropyAnalysisConfig] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model = self._load_model(model_name, device)
        self.config = config or EntropyAnalysisConfig()
        self.logits_entropy_thresholds = {}
        self.attn_entropy_thresholds = {}

    @abstractmethod
    def _load_model(self, model_name: str, device: str):
        pass

    @staticmethod
    def calculate_entropy(probs: np.ndarray) -> float:
        """Calculate the entropy of a probability distribution."""
        probs = probs[probs > 0]
        return entropy(probs, base=2)

    def calculate_sequence_entropy(self, probs: torch.Tensor) -> List[float]:
        """Calculate entropy for each token in a sequence."""
        return [self.calculate_entropy(token_probs.cpu().numpy()) for token_probs in probs]

    def collect_calibration_data(self, input_output_pairs: List[Tuple[str, str]]) -> None:
        """
        Collect entropy data from input-output pairs to calibrate thresholds.

        Args:
            input_output_pairs: List of tuples containing input and corresponding output text.
        """
        logits_entropy_list = []
        attention_entropy_list = []

        for input_text, output_text in input_output_pairs:
            try:
                # Concatenate input and output text for tokenization
                full_text = input_text + output_text
                inputs = self.tokenizer(full_text, return_tensors='pt').to(self.device)
                output_ids = self.tokenizer(output_text, return_tensors='pt')['input_ids'][0].to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=True)

                # Extract logits and attentions for output tokens
                logits = outputs.logits[0, -len(output_ids):, :]  # Shape: [output_len, vocab_size]
                attentions = outputs.attentions  # List of tensors per layer

                # Logits entropy
                probs = F.softmax(logits, dim=-1)
                logits_entropies = [self.calculate_entropy(token_probs.cpu().numpy()) for token_probs in probs]
                logits_entropy_list.extend(logits_entropies)

                # Attention entropy
                last_layer_attentions = attentions[-1][0]  # Shape: [num_heads, seq_len, seq_len]
                output_attn_weights = last_layer_attentions[:, -len(output_ids):,
                                      :]  # Shape: [num_heads, output_len, seq_len]

                # Compute entropy per head and then average over heads for each token
                for token_attn_weights in output_attn_weights.permute(1, 0,
                                                                      2):  # Shape: [output_len, num_heads, seq_len]
                    head_entropies = [self.calculate_entropy(head_weights.cpu().numpy()) for head_weights in
                                      token_attn_weights]
                    token_entropy = np.mean(head_entropies)
                    attention_entropy_list.append(token_entropy)

            except Exception as e:
                logger.error(f"Error processing input-output pair: {e}", exc_info=True)
                continue

        # Use percentiles for threshold calculations
        self.logits_entropy_thresholds['high'] = np.percentile(logits_entropy_list, 75)
        self.logits_entropy_thresholds['low'] = np.percentile(logits_entropy_list, 25)
        self.attn_entropy_thresholds['high'] = np.percentile(attention_entropy_list, 75)
        self.attn_entropy_thresholds['low'] = np.percentile(attention_entropy_list, 25)

        logger.info(
            f"Logits Entropy Thresholds - Low (25th percentile): {self.logits_entropy_thresholds['low']:.4f}, High (75th percentile): {self.logits_entropy_thresholds['high']:.4f}")
        logger.info(
            f"Attention Entropy Thresholds - Low (25th percentile): {self.attn_entropy_thresholds['low']:.4f}, High (75th percentile): {self.attn_entropy_thresholds['high']:.4f}")

    def categorize_state(self, logits_entropy: float, attention_entropy: float) -> str:
        """
        Categorize the model's state based on entropy thresholds.

        Args:
            logits_entropy: Entropy of the logits distribution.
            attention_entropy: Entropy of the attention weights.

        Returns:
            A string indicating the model's state.
        """
        high_logits_entropy = self.logits_entropy_thresholds['high']
        low_logits_entropy = self.logits_entropy_thresholds['low']
        high_attn_entropy = self.attn_entropy_thresholds['high']
        low_attn_entropy = self.attn_entropy_thresholds['low']

        if logits_entropy > high_logits_entropy and attention_entropy > high_attn_entropy:
            return 'Uncertain'
        elif logits_entropy < low_logits_entropy and attention_entropy < low_attn_entropy:
            return 'Overconfident'
        else:
            return 'Confident'

    def analyze_model_state(self, input_text: str) -> Dict:
        """
        Analyze the model's state for the given input text.

        Args:
            input_text: The input text to analyze.

        Returns:
            A dictionary containing analysis results.
        """
        try:
            inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)

            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states

            analysis_results = {}

            if self.config.logits_entropy.enabled:
                probs = F.softmax(logits, dim=-1)
                logits_entropies = self.calculate_sequence_entropy(probs)
                analysis_results['logits_entropies'] = logits_entropies

            if self.config.attention_entropy.enabled:
                # Compute per-head entropies and average across heads
                last_layer_attentions = attentions[-1][0]  # Shape: [num_heads, seq_len, seq_len]
                attention_entropies = []
                for token_attn_weights in last_layer_attentions.permute(1, 0, 2):  # Iterate over tokens
                    head_entropies = [self.calculate_entropy(head_weights.cpu().numpy()) for head_weights in
                                      token_attn_weights]
                    token_entropy = np.mean(head_entropies)
                    attention_entropies.append(token_entropy)
                analysis_results['attention_entropies'] = attention_entropies

            if self.config.mc_dropout.enabled:
                mean_probs, var_probs = self.mc_dropout(inputs)
                uncertainty = var_probs[0, -1, :].mean().item()
                analysis_results['mc_dropout_uncertainty'] = uncertainty

            if self.config.perplexity.enabled:
                perplexity = self.calculate_perplexity(input_text)
                analysis_results['perplexity'] = perplexity

            # Categorize state for last token
            if 'logits_entropies' in analysis_results and 'attention_entropies' in analysis_results:
                logits_entropy_last = analysis_results['logits_entropies'][-1]
                attention_entropy_last = analysis_results['attention_entropies'][-1]
                state = self.categorize_state(logits_entropy_last, attention_entropy_last)
                analysis_results['model_state'] = state

            return analysis_results

        except Exception as e:
            logger.error(f"Error analyzing model state: {e}", exc_info=True)
            return {}

    def analyze_hidden_states(self, hidden_states: List[torch.Tensor]) -> List[Dict]:
        """
        Analyze the mean and standard deviation of activations in each hidden layer.

        Args:
            hidden_states: List of hidden state tensors from each layer.

        Returns:
            A list of dictionaries containing stats per layer.
        """
        layer_stats = []
        for idx, layer_hidden in enumerate(hidden_states):
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

    def analyze_attention_heads(self, attentions: List[torch.Tensor]) -> List[float]:
        """
        Analyze the entropy of attention distributions for each head.

        Args:
            attentions: List of attention tensors from each layer.

        Returns:
            A list of entropy values per head.
        """
        last_layer_attn = attentions[-1][0]  # Shape: [num_heads, seq_len, seq_len]
        num_heads = last_layer_attn.size(0)
        head_entropies = []
        # Iterate over heads
        for head in range(num_heads):
            head_attn = last_layer_attn[head]  # Shape: [seq_len, seq_len]
            token_attn = head_attn[-1]  # Use the last token
            entropy_value = self.calculate_entropy(token_attn.cpu().numpy())
            head_entropies.append(entropy_value)
        return head_entropies

    def mc_dropout(self, inputs: Dict[str, torch.Tensor], n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Monte Carlo dropout to estimate uncertainty.

        Args:
            inputs: Dictionary of inputs for the model.
            n_samples: Number of samples for MC dropout.

        Returns:
            A tuple containing mean and variance of the probabilities.
        """
        self.model.train()
        dropout_outputs = []
        for _ in range(n_samples):
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            dropout_outputs.append(probs)
        self.model.eval()
        probs_stack = torch.stack(dropout_outputs, dim=0)
        return probs_stack.mean(dim=0), probs_stack.var(dim=0)

    def calculate_perplexity(self, input_text: str) -> float:
        """Calculate perplexity of the input text."""
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
        return torch.exp(loss).item()

    @staticmethod
    def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0,
                              filter_value: float = -float('Inf')) -> torch.Tensor:
        """
        Filter logits using top-k and/or top-p (nucleus) filtering.

        Args:
            logits: Logits distribution shape (batch_size, vocab_size).
            top_k: Keep only top k tokens with highest probability.
            top_p: Keep the top tokens with cumulative probability >= top_p.
            filter_value: The value to replace filtered logits with.

        Returns:
            Filtered logits tensor.
        """
        # Top-k filtering
        if top_k > 0:
            top_k = min(max(top_k, 1), logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, filter_value)
        # Top-p (nucleus) filtering
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the index to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits.scatter_(dim=-1, index=indices_to_remove, value=filter_value)
        return logits

    def generate_and_analyze(self, input_text: str, max_length: int = 50, temperature: float = 1.0,
                             top_k: int = 0, top_p: float = 0.0) -> Dict:
        """
        Generate text from the input and analyze model state at each step.

        Args:
            input_text: The input text to start generation.
            max_length: Maximum length of generated text.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.

        Returns:
            A dictionary containing generation results and analyses.
        """
        self.model.eval()
        generated_ids = self.tokenizer(input_text, return_tensors='pt').input_ids.to(self.device)
        past_key_values = None
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info(f"Input Text: {input_text}")

        generation_results = []
        generated_text = input_text

        for step in range(max_length):
            try:
                if past_key_values is not None:
                    input_ids_step = generated_ids[:, -1:]  # Last generated token
                else:
                    input_ids_step = generated_ids

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids_step,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True,
                        output_hidden_states=True
                    )

                past_key_values = outputs.past_key_values
                logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
                attentions = outputs.attentions
                hidden_states = outputs.hidden_states

                step_analysis = self.analyze_step(logits, attentions, hidden_states)
                generation_results.append(step_analysis)

                # Sampling
                next_token_id = self.sample_next_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

                generated_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=False)
                generated_text += generated_token

                logger.info(f"Step {step + 1} - Generated Token: {generated_token}")
                # Print step analysis if needed
                # logger.debug(f"Step Analysis: {step_analysis}")
                logger.info("-" * 50)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    logger.info("End of sequence reached.")
                    break

            except Exception as e:
                logger.error(f"Error in generation step {step + 1}: {e}", exc_info=True)
                break

        logger.info("Final Generated Text:")
        logger.info(generated_text)

        return {
            'generated_ids': generated_ids,
            'generated_text': generated_text,
            'step_analyses': generation_results
        }

    def analyze_step(self, logits: torch.Tensor, attentions: List[torch.Tensor],
                     hidden_states: List[torch.Tensor]) -> Dict:
        """
        Analyze the model's state at a single generation step.

        Args:
            logits: Logits tensor from the model.
            attentions: List of attention tensors.
            hidden_states: List of hidden state tensors.

        Returns:
            A dictionary containing analysis results for the step.
        """
        step_analysis = {}

        if self.config.logits_entropy.enabled:
            probs = F.softmax(logits[:, -1, :], dim=-1)
            logits_entropy = self.calculate_entropy(probs[0].cpu().numpy())
            step_analysis['logits_entropy'] = logits_entropy

        if self.config.attention_entropy.enabled:
            last_layer_attentions = attentions[-1][0]  # Shape: [num_heads, seq_len, seq_len]
            head_entropies = []
            for head_attn_weights in last_layer_attentions:  # Iterate over heads
                attn_weights = head_attn_weights[-1, :]  # Get attention for the last token
                entropy_value = self.calculate_entropy(attn_weights.cpu().numpy())
                head_entropies.append(entropy_value)
            attention_entropy = np.mean(head_entropies)
            step_analysis['attention_entropy'] = attention_entropy

        if 'logits_entropy' in step_analysis and 'attention_entropy' in step_analysis:
            state = self.categorize_state(step_analysis['logits_entropy'], step_analysis['attention_entropy'])
            step_analysis['model_state'] = state

        return step_analysis

    def sample_next_token(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                          top_p: float = 0.0) -> torch.Tensor:
        """
        Sample the next token from the logits.

        Args:
            logits: Logits tensor of shape [batch_size, seq_len, vocab_size].
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.

        Returns:
            Tensor containing the next token ID.
        """
        logits = logits[:, -1, :] / temperature
        filtered_logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        return next_token_id

    def visualize_attention(self, attentions: List[torch.Tensor], input_tokens: List[str]) -> None:
        """
        Visualize the attention weights as a heatmap.

        Args:
            attentions: List of attention tensors from the model.
            input_tokens: List of token strings corresponding to the input.
        """
        if not self.config.attention_entropy.enabled:
            logger.warning("Attention visualization is not enabled in the configuration.")
            return

        try:
            last_layer_attn = attentions[-1][0].mean(dim=0).cpu().numpy()  # Shape: [seq_len, seq_len]

            plt.figure(figsize=(10, 8))
            sns.heatmap(last_layer_attn, xticklabels=input_tokens, yticklabels=input_tokens, cmap='viridis')
            plt.title('Attention Heatmap')
            plt.xlabel('Key Positions')
            plt.ylabel('Query Positions')
            plt.show()
        except Exception as e:
            logger.error(f"Error in attention visualization: {e}", exc_info=True)

    def visualize_entropy_over_time(self, generation_results: Dict):
        """
        Visualize the entropy of logits and attention over the generation steps.

        Args:
            generation_results: Dictionary containing generation results and analyses.
        """
        if not (self.config.logits_entropy.enabled and self.config.attention_entropy.enabled):
            logger.warning("Logits and attention entropy visualization is not enabled in the configuration.")
            return

        steps = range(1, len(generation_results['step_analyses']) + 1)
        logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
        attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

        plt.figure(figsize=(12, 6))
        plt.plot(steps, logits_entropies, label='Logits Entropy', marker='o')
        plt.plot(steps, attention_entropies, label='Attention Entropy', marker='s')
        plt.xlabel('Generation Step')
        plt.ylabel('Entropy')
        plt.title('Entropy Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_model_states(self, generation_results: Dict):
        """
        Visualize the model's state over the generation steps.

        Args:
            generation_results: Dictionary containing generation results and analyses.
        """
        if not generation_results['step_analyses']:
            logger.warning("No step analyses available for visualization.")
            return

        if 'model_state' not in generation_results['step_analyses'][0]:
            logger.warning("Model state visualization is not available.")
            return

        steps = range(1, len(generation_results['step_analyses']) + 1)
        states = [step['model_state'] for step in generation_results['step_analyses']]

        state_to_num = {'Uncertain': 0, 'Confident': 1, 'Overconfident': 2}
        numeric_states = [state_to_num[state] for state in states]

        plt.figure(figsize=(12, 6))
        plt.plot(steps, numeric_states, marker='o')
        plt.yticks([0, 1, 2], ['Uncertain', 'Confident', 'Overconfident'])
        plt.xlabel('Generation Step')
        plt.ylabel('Model State')
        plt.title('Model State Over Time')
        plt.grid(True)
        plt.show()

    def visualize_entropy_distribution(self, generation_results: Dict):
        logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
        attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(logits_entropies, kde=True)
        plt.title('Distribution of Logits Entropy')
        plt.xlabel('Entropy')

        plt.subplot(1, 2, 2)
        sns.histplot(attention_entropies, kde=True)
        plt.title('Distribution of Attention Entropy')
        plt.xlabel('Entropy')

        plt.tight_layout()
        plt.show()

    def visualize_attention_head_entropy(self, attentions: List[torch.Tensor]):
        last_layer_attn = attentions[-1][0]  # Shape: [num_heads, seq_len, seq_len]
        num_heads, seq_len, _ = last_layer_attn.shape

        head_entropies = np.zeros((num_heads, seq_len))
        for head in range(num_heads):
            for token in range(seq_len):
                head_entropies[head, token] = self.calculate_entropy(last_layer_attn[head, token].cpu().numpy())

        plt.figure(figsize=(12, 8))
        sns.heatmap(head_entropies, cmap='viridis')
        plt.title('Attention Head Entropy')
        plt.xlabel('Token Position')
        plt.ylabel('Attention Head')
        plt.show()

    def rolling_entropy(self, entropies: List[float], window: int = 5):
        return pd.Series(entropies).rolling(window=window).mean().tolist()

    def visualize_rolling_entropy(self, generation_results: Dict, window: int = 5):
        steps = range(1, len(generation_results['step_analyses']) + 1)
        logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
        attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

        rolling_logits = self.rolling_entropy(logits_entropies, window)
        rolling_attention = self.rolling_entropy(attention_entropies, window)

        plt.figure(figsize=(12, 6))
        plt.plot(steps[window-1:], rolling_logits[window-1:], label='Rolling Logits Entropy', marker='o')
        plt.plot(steps[window-1:], rolling_attention[window-1:], label='Rolling Attention Entropy', marker='s')
        plt.xlabel('Generation Step')
        plt.ylabel('Rolling Entropy')
        plt.title(f'Rolling Entropy Over Time (Window = {window})')
        plt.legend()
        plt.grid(True)
        plt.show()

    def entropy_gradient(self, entropies: List[float]):
        return np.gradient(entropies)

    def visualize_entropy_gradient(self, generation_results: Dict):
        steps = range(1, len(generation_results['step_analyses']))
        logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
        attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

        logits_gradient = self.entropy_gradient(logits_entropies)
        attention_gradient = self.entropy_gradient(attention_entropies)

        plt.figure(figsize=(12, 6))
        plt.plot(steps, logits_gradient, label='Logits Entropy Gradient', marker='o')
        plt.plot(steps, attention_gradient, label='Attention Entropy Gradient', marker='s')
        plt.xlabel('Generation Step')
        plt.ylabel('Entropy Gradient')
        plt.title('Entropy Gradient Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def analyze_entropy_correlation(self, generation_results: Dict):
        logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
        attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

        correlation = np.corrcoef(logits_entropies, attention_entropies)[0, 1]

        plt.figure(figsize=(8, 6))
        plt.scatter(logits_entropies, attention_entropies)
        plt.xlabel('Logits Entropy')
        plt.ylabel('Attention Entropy')
        plt.title(f'Logits vs Attention Entropy (Correlation: {correlation:.2f})')
        plt.grid(True)
        plt.show()

        return correlation

    def analyze_entropy_thresholds(self, generation_results: Dict, logits_threshold: float, attention_threshold: float):
        steps = range(1, len(generation_results['step_analyses']) + 1)
        logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
        attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

        logits_above_threshold = [1 if e > logits_threshold else 0 for e in logits_entropies]
        attention_above_threshold = [1 if e > attention_threshold else 0 for e in attention_entropies]

        plt.figure(figsize=(12, 6))
        plt.step(steps, logits_above_threshold, label='Logits Entropy', where='post')
        plt.step(steps, attention_above_threshold, label='Attention Entropy', where='post')
        plt.xlabel('Generation Step')
        plt.ylabel('Above Threshold')
        plt.title(f'Entropy Threshold Analysis (Logits: {logits_threshold}, Attention: {attention_threshold})')
        plt.legend()
        plt.ylim(-0.1, 1.1)
        plt.grid(True)
        plt.show()

class BasicEntropyAnalysisWrapper(BaseEntropyAnalysisWrapper):
    def _load_model(self, model_name: str, device: str) -> AutoModelForCausalLM:
        """Load the causal language model with required configurations."""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            return_dict_in_generate=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        return model.to(device)


if __name__ == "__main__":
    config = EntropyAnalysisConfig()

    wrapper = BasicEntropyAnalysisWrapper("meta-llama/Llama-3.2-3B-Instruct", config=config)

    # Collect calibration data
    input_output_pairs = [
        (
            """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are an AI assistant created to be helpful and honest. Always think step by step and layout your chain of thought in great detail.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What's the capital of France?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
            """
The capital of France is Paris.
<|eot_id|>""",
        ),
        (
            """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are an AI assistant created to be helpful and honest. Always think step by step and layout your chain of thought in great detail.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Explain the concept of photosynthesis.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
            """
Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy. It involves the following steps:

1. Light absorption by chlorophyll in plant leaves
2. Conversion of light energy to chemical energy (ATP and NADPH)
3. Use of this energy to convert CO2 and water into glucose
4. Release of oxygen as a byproduct

This process is crucial for life on Earth as it produces oxygen and forms the base of most food chains.
<|eot_id|>""",
        ),
        (
            """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are an AI assistant created to be helpful and honest. Always think step by step and layout your chain of thought in great detail.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write a haiku about spring.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
            """
Here's a haiku about spring:

Soft breeze whispers by
Cherry blossoms paint the sky
New life awakens
<|eot_id|>""",
        ),
        (
            """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are an AI assistant created to be helpful and honest. Always think step by step and layout your chain of thought in great detail.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What are the three states of matter?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
            """
The three main states of matter are:

1. Solid
2. Liquid
3. Gas

Each state has distinct properties based on the arrangement and movement of particles within the substance.
<|eot_id|>""",
        ),
        (
            """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are an AI assistant created to be helpful and honest. Always think step by step and layout your chain of thought in great detail.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

How does a rainbow form?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
            """
A rainbow forms through the following process:

1. Sunlight enters water droplets in the air
2. Light is refracted (bent) as it enters the droplet
3. The light is then reflected off the back of the droplet
4. As it exits, it's refracted again and separated into different wavelengths
5. This separation creates the spectrum of colors we see
6. The observer must be at the correct angle between the sun and water droplets to see the rainbow

Rainbows appear as an arc because of the specific angle at which this light refraction occurs.
<|eot_id|>""",
        ),
    ]
    wrapper.collect_calibration_data(input_output_pairs)

    # Generate and analyze text
    generation_input = """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are an AI assistant created to be helpful and honest. Always think step by step and layout your chain of thought in great detail.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

How many r's are in the word strawberry?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    generation_results = wrapper.generate_and_analyze(
        generation_input,
        max_length=200,
        temperature=0.3,
    )

    # Visualize results
    wrapper.visualize_entropy_over_time(generation_results)
    wrapper.visualize_model_states(generation_results)
    wrapper.visualize_entropy_over_time(generation_results)
    wrapper.visualize_model_states(generation_results)
    wrapper.visualize_entropy_distribution(generation_results)
    wrapper.visualize_rolling_entropy(generation_results)
    wrapper.visualize_entropy_gradient(generation_results)
    correlation = wrapper.analyze_entropy_correlation(generation_results)
    wrapper.analyze_entropy_thresholds(generation_results, logits_threshold=2.0, attention_threshold=1.5)