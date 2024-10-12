import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod


class ConfigurableAnalysis:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled


class EntropyAnalysisConfig:
    def __init__(self):
        self.logits_entropy = ConfigurableAnalysis()
        self.attention_entropy = ConfigurableAnalysis()
        self.hidden_states = ConfigurableAnalysis()
        self.head_entropies = ConfigurableAnalysis()
        self.mc_dropout = ConfigurableAnalysis()
        self.gradient_importance = ConfigurableAnalysis()
        self.perplexity = ConfigurableAnalysis()
        self.layer_wise_activation = ConfigurableAnalysis()


class BaseEntropyAnalysisWrapper(ABC):
    def __init__(self, model_name: str, device: str = "cuda", config: Optional[EntropyAnalysisConfig] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = self._load_model(model_name, device)
        self.device = device
        self.config = config or EntropyAnalysisConfig()
        self.logits_mean = None
        self.logits_std = None
        self.attn_mean = None
        self.attn_std = None

    @abstractmethod
    def _load_model(self, model_name: str, device: str):
        pass

    @staticmethod
    def calculate_entropy(probs: np.ndarray) -> float:
        probs = probs[probs > 0]
        return entropy(probs, base=2)

    def calculate_sequence_entropy(self, probs: torch.Tensor) -> List[float]:
        return [self.calculate_entropy(token_probs[token_probs > 0].cpu().numpy()) for token_probs in probs]

    def collect_calibration_data(self, input_output_pairs: List[Tuple[str, str]]) -> None:
        logits_entropy_list = []
        attention_entropy_list = []

        for input_text, output_text in input_output_pairs:
            try:
                full_text = input_text + output_text
                inputs = self.tokenizer(full_text, return_tensors='pt').to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=True)

                logits = outputs.logits
                attentions = outputs.attentions

                probs = F.softmax(logits, dim=-1)
                sequence_probs = probs[0]
                logits_entropies = self.calculate_sequence_entropy(sequence_probs)
                logits_entropy_list.extend(logits_entropies)

                last_layer_attentions = attentions[-1][0].mean(dim=0)
                attention_entropies = [self.calculate_entropy(attn_weights.cpu().numpy())
                                       for attn_weights in last_layer_attentions]
                attention_entropy_list.extend(attention_entropies)

                print(f"Input: {input_text}")
                print(f"Output: {output_text}")
                print(f"Logits Entropies: {logits_entropies}")
                print(f"Attention Entropies: {attention_entropies}")
                print("-" * 50)

            except Exception as e:
                print(f"Error processing input-output pair: {e}")

        self.logits_mean = np.mean(logits_entropy_list)
        self.logits_std = np.std(logits_entropy_list)
        self.attn_mean = np.mean(attention_entropy_list)
        self.attn_std = np.std(attention_entropy_list)

        print(f"Logits Entropy - Mean: {self.logits_mean:.4f}, Std: {self.logits_std:.4f}")
        print(f"Attention Entropy - Mean: {self.attn_mean:.4f}, Std: {self.attn_std:.4f}")
        print("=" * 50)

    def categorize_state(self, logits_entropy: float, attention_entropy: float) -> str:
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

    def analyze_model_state(self, input_text: str, visualize: bool = False) -> Dict:
        try:
            inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)

            logits = outputs.logits
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states

            analysis_results = {}

            if self.config.logits_entropy.enabled:
                probs = F.softmax(logits, dim=-1)
                sequence_probs = probs[0]
                logits_entropies = self.calculate_sequence_entropy(sequence_probs)
                analysis_results['logits_entropies'] = logits_entropies

            if self.config.attention_entropy.enabled:
                last_layer_attentions = attentions[-1][0].mean(dim=0)
                attention_entropies = [self.calculate_entropy(attn_weights.cpu().numpy())
                                       for attn_weights in last_layer_attentions]
                analysis_results['attention_entropies'] = attention_entropies

            if self.config.hidden_states.enabled:
                hidden_state_stats = self.analyze_hidden_states(hidden_states)
                analysis_results['hidden_state_stats'] = hidden_state_stats

                if visualize:
                    self.plot_hidden_state_stats(hidden_state_stats)

            if self.config.head_entropies.enabled:
                head_entropies = self.analyze_attention_heads(attentions)
                analysis_results['head_entropies'] = head_entropies

            if self.config.mc_dropout.enabled:
                mean_probs, var_probs = self.mc_dropout(inputs)
                uncertainty = var_probs[0, -1, :].mean().item()
                analysis_results['mc_dropout_uncertainty'] = uncertainty

            if self.config.layer_wise_activation.enabled:
                layer_stats = self.layer_wise_activation_stats(hidden_states)
                analysis_results['layer_wise_activation_stats'] = layer_stats

                if visualize:
                    self.plot_layer_wise_activation_stats(layer_stats)

            if self.config.perplexity.enabled:
                perplexity = self.calculate_perplexity(input_text)
                analysis_results['perplexity'] = perplexity

            # Categorize state for last token
            if 'logits_entropies' in analysis_results and 'attention_entropies' in analysis_results:
                logits_entropy_last = analysis_results['logits_entropies'][-1]
                attention_entropy_last = analysis_results['attention_entropies'][-1]
                state = self.categorize_state(logits_entropy_last, attention_entropy_last)
                analysis_results['model_state'] = state

            if visualize and self.config.attention_entropy.enabled:
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                self.visualize_attention(attentions, tokens)

            if visualize and self.config.gradient_importance.enabled:
                self.gradient_importance(input_text)

            return analysis_results

        except Exception as e:
            print(f"Error analyzing model state: {e}")
            return {}

    def analyze_hidden_states(self, hidden_states: List[torch.Tensor]) -> List[Dict]:
        layer_stats = []
        for idx, layer_hidden in enumerate(hidden_states):
            mean_activation = layer_hidden.mean().item()
            std_activation = layer_hidden.std().item()
            layer_stats.append({
                'layer': idx,
                'mean_activation': mean_activation,
                'std_activation': std_activation
            })
        return layer_stats

    def plot_hidden_state_stats(self, hidden_state_stats: List[Dict]) -> None:
        layers = [stat['layer'] for stat in hidden_state_stats]
        mean_activations = [stat['mean_activation'] for stat in hidden_state_stats]
        std_activations = [stat['std_activation'] for stat in hidden_state_stats]

        plt.figure()
        plt.plot(layers, mean_activations, marker='o', label='Mean Activation')
        plt.plot(layers, std_activations, marker='x', label='Std Activation')
        plt.title('Hidden State Activation Stats')
        plt.xlabel('Layer')
        plt.ylabel('Activation Value')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def analyze_attention_heads(self, attentions: List[torch.Tensor]) -> List[float]:
        last_layer_attn = attentions[-1][0]
        num_heads = last_layer_attn.size(0)
        return [self.calculate_entropy(last_layer_attn[head][-1, :].cpu().numpy()) for head in range(num_heads)]

    def mc_dropout(self, inputs: Dict[str, torch.Tensor], n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def layer_wise_activation_stats(self, hidden_states: List[torch.Tensor]) -> List[Dict]:
        return [{
            'layer': idx,
            'mean_activation': layer_hidden.mean().item(),
            'std_activation': layer_hidden.std().item(),
            'norm_activation': layer_hidden.norm().item()
        } for idx, layer_hidden in enumerate(hidden_states)]

    def plot_layer_wise_activation_stats(self, layer_stats: List[Dict[str, float]]) -> None:
        layers = [stat['layer'] for stat in layer_stats]
        mean_activations = [stat['mean_activation'] for stat in layer_stats]
        std_activations = [stat['std_activation'] for stat in layer_stats]
        norms = [stat.get('norm_activation', None) for stat in layer_stats]

        plt.figure()
        plt.plot(layers, mean_activations, marker='o', label='Mean Activation')
        plt.plot(layers, std_activations, marker='x', label='Std Activation')
        if all(norm is not None for norm in norms):
            plt.plot(layers, norms, marker='s', label='Activation Norm')
        plt.title('Layer-wise Activation Stats')
        plt.xlabel('Layer')
        plt.ylabel('Activation Value')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def calculate_perplexity(self, input_text: str) -> float:
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
        return torch.exp(loss).item()

    @staticmethod
    def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0,
                              filter_value: float = -float('Inf')) -> torch.Tensor:
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(1)
            logits = torch.where(logits < min_values, torch.full_like(logits, filter_value), logits)
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits.scatter_(dim=-1, index=indices_to_remove, value=filter_value)
        return logits

    def build_attention_mask(self, input_ids: torch.Tensor, past_length: int) -> torch.Tensor:
        """
        Build an attention mask for the current generation step.

        Args:
        input_ids (torch.Tensor): The input token IDs.
        past_length (int): The length of the past context.

        Returns:
        torch.Tensor: The attention mask.
        """
        # Get the length of the current input
        input_length = input_ids.size(1)

        # Create a tensor of ones with shape (batch_size, past_length + input_length)
        mask = torch.ones((input_ids.size(0), past_length + input_length), dtype=torch.long, device=input_ids.device)

        # If there's a past context, set those positions to 1 (attend to all past tokens)
        if past_length > 0:
            mask[:, :past_length] = 1

        # Set padding token positions to 0 in the mask for the current input
        mask[:, past_length:][input_ids == self.tokenizer.pad_token_id] = 0

        return mask

    def generate_and_analyze(self, input_text: str, max_length: int = 50, method: str = 'temperature',
                             plot_metrics: bool = False, **kwargs) -> Dict:
        self.model.eval()
        input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids.to(self.device)
        generated_ids = input_ids.clone()
        past_key_values = None
        past_length = 0
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"Input Text: {input_text}")

        generation_results = []

        for step in range(max_length):
            # Build the attention mask for the current step
            attention_mask = self.build_attention_mask(generated_ids, past_length)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids[:, -1:] if past_key_values is not None else generated_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=True
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states

            step_analysis = self.analyze_step(logits, attentions, hidden_states)
            generation_results.append(step_analysis)

            next_token_id = self.sample_next_token(logits, method, **kwargs)
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            # Update past_length after generating a new token
            past_length = generated_ids.size(1) - 1

            generated_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            print(f"Step {step + 1} - Generated Token: {generated_token}")
            print(f"Step Analysis: {step_analysis}")
            print("-" * 50)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                print("End of sequence reached.")
                break

        final_generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("Final Generated Text:")
        print(final_generated_text)

        if plot_metrics:
            self.plot_generation_metrics(generation_results)

        return {
            'generated_text': final_generated_text,
            'step_analyses': generation_results
        }

    def analyze_step(self, logits: torch.Tensor, attentions: List[torch.Tensor],
                     hidden_states: List[torch.Tensor]) -> Dict:
        step_analysis = {}

        if self.config.logits_entropy.enabled:
            probs = F.softmax(logits[:, -1, :], dim=-1)
            logits_entropy = self.calculate_entropy(probs[0].cpu().numpy())
            step_analysis['logits_entropy'] = logits_entropy

        if self.config.attention_entropy.enabled:
            last_layer_attentions = attentions[-1][0]
            attn_weights = last_layer_attentions[:, -1, :].mean(dim=0)
            attention_entropy = self.calculate_entropy(attn_weights.cpu().numpy())
            step_analysis['attention_entropy'] = attention_entropy

        if self.config.hidden_states.enabled:
            last_hidden_state = hidden_states[-1]
            hidden_state_norm = last_hidden_state.norm().item()
            step_analysis['hidden_state_norm'] = hidden_state_norm

        if 'logits_entropy' in step_analysis and 'attention_entropy' in step_analysis:
            state = self.categorize_state(step_analysis['logits_entropy'], step_analysis['attention_entropy'])
            step_analysis['model_state'] = state

        return step_analysis

    def sample_next_token(self, logits: torch.Tensor, method: str, **kwargs) -> torch.Tensor:
        if method == 'temperature':
            temperature = kwargs.get('temperature', 1.0)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        elif method == 'top-k':
            k = kwargs.get('k', 50)
            logits = logits[:, -1, :]
            filtered_logits = self.top_k_top_p_filtering(logits, top_k=k)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        elif method == 'top-p':
            p = kwargs.get('p', 0.9)
            logits = logits[:, -1, :]
            filtered_logits = self.top_k_top_p_filtering(logits, top_p=p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return next_token_id

    def plot_generation_metrics(self, generation_results: List[Dict], metrics: Optional[List[str]] = None) -> None:
        """
        Plot specified metrics over the generation steps.
        Args:
            generation_results (List[Dict]): List of step_analysis dictionaries.
            metrics (List[str]): List of metric names to plot. If None, plot all available metrics.
        """
        available_metrics = set()
        for step in generation_results:
            available_metrics.update(step.keys())
        if metrics is None:
            metrics = list(available_metrics)
        else:
            metrics = [metric for metric in metrics if metric in available_metrics]

        steps = range(len(generation_results))

        for metric in metrics:
            values = [step.get(metric) for step in generation_results]
            if all(v is not None for v in values):
                plt.figure()
                plt.plot(steps, values, marker='o')
                plt.title(f'{metric} over time')
                plt.xlabel('Generation Step')
                plt.ylabel(metric)
                plt.tight_layout()
                plt.show()
            else:
                print(f"Metric {metric} not available for all steps.")

    def visualize_attention(self, attentions: List[torch.Tensor], input_tokens: List[str]) -> None:
        if not self.config.attention_entropy.enabled:
            print("Attention visualization is not enabled in the configuration.")
            return

        try:
            last_layer_attn = attentions[-1][0]  # [num_heads, seq_len, seq_len]
            avg_attn = last_layer_attn.mean(dim=0)  # [seq_len, seq_len]

            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_attn.cpu().numpy(), xticklabels=input_tokens, yticklabels=input_tokens)
            plt.title('Attention Heatmap')
            plt.xlabel('Key Positions')
            plt.ylabel('Query Positions')
            plt.show()
        except Exception as e:
            print(f"Error in attention visualization: {e}")

    def gradient_importance(self, input_text: str) -> None:
        if not self.config.gradient_importance.enabled:
            print("Gradient importance analysis is not enabled in the configuration.")
            return

        try:
            inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
            self.model.zero_grad()
            embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
            embeddings.retain_grad()
            outputs = self.model(inputs_embeds=embeddings)
            loss = outputs.logits[:, -1, :].mean()
            loss.backward()
            gradients = embeddings.grad
            gradient_importance = gradients.abs().sum(dim=-1).squeeze().cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            plt.figure(figsize=(12, 6))
            plt.bar(range(len(tokens)), gradient_importance)
            plt.xticks(range(len(tokens)), tokens, rotation='vertical')
            plt.title('Gradient Importance')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in gradient importance analysis: {e}")


class BasicEntropyAnalysisWrapper(BaseEntropyAnalysisWrapper):
    def _load_model(self, model_name: str, device: str) -> AutoModelForCausalLM:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            return_dict_in_generate=True,
            output_attentions=True,
            output_hidden_states=True
        ).to(device)


# Example usage
if __name__ == "__main__":
    config = EntropyAnalysisConfig()
    config.gradient_importance.enabled = True
    config.layer_wise_activation.enabled = True
    config.attention_entropy.enabled = True
    config.hidden_states.enabled = True

    wrapper = BasicEntropyAnalysisWrapper('gpt2', config=config, device='cuda')

    # Collect calibration data
    input_output_pairs = [
        ("Once upon a time", " there was a brave knight."),
        ("The capital of France is", " Paris."),
        ("In quantum mechanics,", " particles can be in multiple states."),
    ]
    wrapper.collect_calibration_data(input_output_pairs)

    # Analyze model state with visualization
    analysis_results = wrapper.analyze_model_state("The quick brown fox", visualize=True)
    print("Analysis Results:")
    print(analysis_results)

    # Generate and analyze text with metric plots
    generation_results = wrapper.generate_and_analyze(
        "The meaning of life is",
        max_length=20,
        method='temperature',
        temperature=0.7,
        plot_metrics=True
    )
    print("Generation Results:")
    print(generation_results)
