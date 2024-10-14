import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from contextlib import contextmanager
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class ConfigurableAnalysis:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled


class EntropyAnalysisConfig:
    def __init__(self):
        self.logits_entropy = ConfigurableAnalysis()
        self.attention_entropy = ConfigurableAnalysis()
        self.mc_dropout = ConfigurableAnalysis()
        self.perplexity = ConfigurableAnalysis()


class BaseEntropyAnalysisWrapper(ABC):
    def __init__(
            self,
            model_name: str,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            config: Optional[EntropyAnalysisConfig] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, clean_up_tokenization_spaces=True
        )
        self.model = self._load_model(model_name, device)
        self.device = device
        self.config = config or EntropyAnalysisConfig()
        self.logits_entropy_list = []
        self.attn_entropy_list = []
        self.classifier = RandomForestClassifier()
        self.scaler = StandardScaler()
        self.calibrated = False  # Indicates if calibration has been performed

    @abstractmethod
    def _load_model(self, model_name: str, device: str):
        pass

    @staticmethod
    def calculate_entropy(probs: np.ndarray) -> float:
        """Calculate Shannon entropy with improved numerical stability."""
        probs = probs[probs > 0]
        log_probs = np.log(probs)
        return -np.sum(probs * log_probs) / np.log(2)

    def calculate_sequence_entropy(self, probs: torch.Tensor) -> np.ndarray:
        """Calculate entropy for each token in the sequence."""
        probs = probs.clone()
        probs[probs <= 0] = 1e-12  # Prevent log(0)
        log_probs = torch.log(probs)
        entropy_values = -(probs * log_probs).sum(dim=-1) / np.log(2)
        return entropy_values.cpu().numpy()

    def calculate_attention_entropy(self, attentions: List[torch.Tensor]) -> List[float]:
        """Calculate entropy for each attention head individually."""
        last_layer_attentions = attentions[-1][0]  # [num_heads, seq_len, seq_len]
        attention_entropies = []
        for head in last_layer_attentions:
            # Calculate entropy for the last query position
            attn_weights = head[-1, :].cpu().numpy()
            attn_entropy = self.calculate_entropy(attn_weights)
            attention_entropies.append(attn_entropy)
        return attention_entropies

    def collect_calibration_data(
            self, input_output_pairs: List[Tuple[str, str]]
    ) -> None:
        """Collect entropy data from calibration input-output pairs."""
        for idx, (input_text, output_text) in enumerate(input_output_pairs):
            try:
                # Tokenize input and output separately
                inputs_input = self.tokenizer(
                    input_text, return_tensors="pt"
                ).to(self.device)
                inputs_output = self.tokenizer(
                    output_text, return_tensors="pt"
                ).to(self.device)

                # Concatenate input_ids and attention_mask
                input_ids = torch.cat(
                    [inputs_input["input_ids"], inputs_output["input_ids"]], dim=1
                )
                attention_mask = torch.cat(
                    [inputs_input["attention_mask"], inputs_output["attention_mask"]], dim=1
                )

                # Determine where the output starts
                output_start = inputs_input["input_ids"].size(1)

                # Prepare inputs for the model
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=True)

                logits = outputs.logits
                attentions = outputs.attentions

                # Calculate probabilities and entropies
                probs = F.softmax(logits, dim=-1)
                sequence_probs = probs[0]
                output_probs = sequence_probs[output_start:]
                logits_entropies = self.calculate_sequence_entropy(output_probs)
                self.logits_entropy_list.extend(logits_entropies)

                attention_entropies = self.calculate_attention_entropy(attentions)
                self.attn_entropy_list.extend(attention_entropies)

                print(f"Calibration Sample {idx + 1}:")
                print(f"Input: {input_text}")
                print(f"Output: {output_text}")
                print(f"Logits Entropies: {logits_entropies}")
                print(f"Attention Entropies: {attention_entropies}")
                print("-" * 50)

            except (KeyError, ValueError) as e:
                print(f"Error processing input-output pair {idx + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error processing pair {idx + 1}: {e}")

        if self.logits_entropy_list and self.attn_entropy_list:
            # Compute robust statistical measures
            self.logits_mean = np.mean(self.logits_entropy_list)
            self.logits_std = np.std(self.logits_entropy_list)
            self.attn_mean = np.mean(self.attn_entropy_list)
            self.attn_std = np.std(self.attn_entropy_list)

            # Alternatively, compute percentiles for thresholds
            self.logits_p25 = np.percentile(self.logits_entropy_list, 25)
            self.logits_p75 = np.percentile(self.logits_entropy_list, 75)
            self.attn_p25 = np.percentile(self.attn_entropy_list, 25)
            self.attn_p75 = np.percentile(self.attn_entropy_list, 75)

            print(f"Calibration Completed:")
            print(f"Logits Entropy - Mean: {self.logits_mean:.4f}, Std: {self.logits_std:.4f}")
            print(
                f"Logits Entropy - 25th Percentile: {self.logits_p25:.4f}, 75th Percentile: {self.logits_p75:.4f}"
            )
            print(f"Attention Entropy - Mean: {self.attn_mean:.4f}, Std: {self.attn_std:.4f}")
            print(
                f"Attention Entropy - 25th Percentile: {self.attn_p25:.4f}, 75th Percentile: {self.attn_p75:.4f}"
            )
            print("=" * 50)

            # Prepare data for classifier training
            features = np.array(
                list(
                    zip(self.logits_entropy_list, self.attn_entropy_list)
                )
            )
            labels = self.assign_labels(self.logits_entropy_list, self.attn_entropy_list)
            # Scale features
            self.scaler.fit(features)
            scaled_features = self.scaler.transform(features)
            # Train classifier
            self.classifier.fit(scaled_features, labels)
            self.calibrated = True
            print("Classifier trained on calibration data.")
            print("=" * 50)
        else:
            print("Calibration data is insufficient for statistical measures.")
            print("=" * 50)

    def assign_labels(
            self, logits_entropies: List[float], attn_entropies: List[float]
    ) -> List[str]:
        """Assign labels based on entropies using percentile thresholds."""
        labels = []
        for log_ent, attn_ent in zip(logits_entropies, attn_entropies):
            if log_ent > self.logits_p75 and attn_ent > self.attn_p75:
                labels.append("Uncertain")
            elif log_ent < self.logits_p25 and attn_ent < self.attn_p25:
                labels.append("Overconfident")
            else:
                labels.append("Confident")
        return labels

    def categorize_state_ml(self, logits_entropy: float, attention_entropy: float) -> str:
        """Categorize model state using the trained classifier."""
        if not self.calibrated:
            raise ValueError("Model is not calibrated. Please run calibration first.")

        feature = np.array([[logits_entropy, attention_entropy]])
        scaled_feature = self.scaler.transform(feature)
        return self.classifier.predict(scaled_feature)[0]

    def categorize_state(
            self, logits_entropy: float, attention_entropy: float
    ) -> str:
        """Categorize model state based on z-scores."""
        z_logits = (logits_entropy - self.logits_mean) / self.logits_std
        z_attn = (attention_entropy - self.attn_mean) / self.attn_std

        if z_logits > 1 and z_attn > 1:
            return "Uncertain"
        elif z_logits < -1 and z_attn < -1:
            return "Overconfident"
        else:
            return "Confident"

    def analyze_model_state(self, input_text: str) -> Dict:
        """Analyze the model's state for a given input."""
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)

            logits = outputs.logits
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states

            analysis_results = {}

            if self.config.logits_entropy.enabled:
                probs = F.softmax(logits, dim=-1)
                logits_entropies = self.calculate_sequence_entropy(probs[0])
                analysis_results["logits_entropies"] = logits_entropies

            if self.config.attention_entropy.enabled:
                attention_entropies = self.calculate_attention_entropy(attentions)
                analysis_results["attention_entropies"] = attention_entropies

            if self.config.mc_dropout.enabled:
                mean_probs, var_probs = self.mc_dropout(inputs)
                uncertainty = var_probs[0, -1, :].mean().item()
                analysis_results["mc_dropout_uncertainty"] = uncertainty

            if self.config.perplexity.enabled:
                perplexity = self.calculate_perplexity(input_text)
                analysis_results["perplexity"] = perplexity

            # Categorize state for the last token
            if (
                    "logits_entropies" in analysis_results
                    and "attention_entropies" in analysis_results
            ):
                logits_entropy_last = analysis_results["logits_entropies"][-1]
                attention_entropy_last = analysis_results["attention_entropies"][-1]
                if self.calibrated:
                    state = self.categorize_state_ml(
                        logits_entropy_last, attention_entropy_last
                    )
                else:
                    state = self.categorize_state(
                        logits_entropy_last, attention_entropy_last
                    )
                analysis_results["model_state"] = state

            return analysis_results

        except (KeyError, ValueError) as e:
            print(f"Error analyzing model state: {e}")
            return {}
        except Exception as e:
            print(f"Unexpected error analyzing model state: {e}")
            return {}

    def analyze_hidden_states(self, hidden_states: List[torch.Tensor]) -> List[Dict]:
        """Analyze hidden states to compute activation statistics."""
        layer_stats = []
        for idx, layer_hidden in enumerate(hidden_states):
            mean_activation = layer_hidden.mean(dim=[0, 1, 2]).item()
            std_activation = layer_hidden.std(dim=[0, 1, 2]).item()
            layer_stats.append(
                {
                    "layer": idx,
                    "mean_activation": mean_activation,
                    "std_activation": std_activation,
                }
            )
        return layer_stats

    def analyze_attention_heads(self, attentions: List[torch.Tensor]) -> List[float]:
        """Analyze attention heads to compute their entropy."""
        last_layer_attn = attentions[-1][0]  # [num_heads, seq_len, seq_len]
        return [
            self.calculate_entropy(attn_weights.cpu().numpy())
            for attn_weights in last_layer_attn
        ]

    @contextmanager
    def enable_dropout(self):
        """Context manager to enable dropout layers during evaluation."""
        self.model.train()
        try:
            yield
        finally:
            self.model.eval()

    def mc_dropout(
            self, inputs: Dict[str, torch.Tensor], n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform Monte Carlo Dropout to estimate uncertainty."""
        dropout_outputs = []
        with self.enable_dropout():
            for _ in range(n_samples):
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                dropout_outputs.append(probs)
        probs_stack = torch.stack(dropout_outputs, dim=0)
        return probs_stack.mean(dim=0), probs_stack.var(dim=0)

    def calculate_perplexity(self, input_text: str) -> float:
        """Calculate perplexity for a given input text."""
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return torch.exp(loss).item()

    @staticmethod
    def top_k_top_p_filtering(
            logits: torch.Tensor,
            top_k: int = 0,
            top_p: float = 0.0,
            filter_value: float = -float("Inf"),
    ) -> torch.Tensor:
        """Filter logits using top-k and/or nucleus (top-p) filtering."""
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(1)
            logits = torch.where(
                logits < min_values, torch.full_like(logits, filter_value), logits
            )
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                ..., :-1
                                                ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits.scatter_(dim=-1, index=indices_to_remove, value=filter_value)
        return logits

    def build_attention_mask(
            self, input_ids: torch.Tensor, past_length: int
    ) -> torch.Tensor:
        """
        Build an attention mask for the current generation step.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            past_length (int): The length of the past context.

        Returns:
            torch.Tensor: The attention mask.
        """
        input_length = input_ids.size(1)
        mask = torch.ones(
            (input_ids.size(0), past_length + input_length),
            dtype=torch.long,
            device=input_ids.device,
        )
        if past_length > 0:
            mask[:, :past_length] = 1
        mask[:, past_length:][input_ids == self.tokenizer.pad_token_id] = 0
        return mask

    def generate_and_analyze(
            self,
            input_text: str,
            max_length: int = 50,
            method: str = "temperature",
            **kwargs,
    ) -> Dict:
        """Generate text and analyze the model's state at each generation step."""
        self.model.eval()
        input_ids = self.tokenizer(
            input_text, return_tensors="pt"
        ).input_ids.to(self.device)
        generated_ids = input_ids.clone()
        past_key_values = None
        past_length = 0
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"Input Text: {input_text}")

        generation_results = []

        for step in range(max_length):
            try:
                # Build the attention mask for the current step
                attention_mask = self.build_attention_mask(generated_ids, past_length)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=generated_ids[:, -1:]
                        if past_key_values is not None
                        else generated_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        use_cache=True,
                        output_attentions=True,
                        output_hidden_states=True,
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
                past_length += 1

                generated_token = self.tokenizer.decode(
                    next_token_id[0], skip_special_tokens=False
                )
                print(f"Step {step + 1} - Generated Token: {generated_token}")
                print("-" * 50)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    print("End of sequence reached.")
                    break

            except (KeyError, ValueError) as e:
                print(f"Error in generation step {step + 1}: {e}")
                break
            except Exception as e:
                print(f"Unexpected error in generation step {step + 1}: {e}")
                break

        final_generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=False
        )
        print("Final Generated Text:")
        print(final_generated_text)

        return {
            "generated_ids": generated_ids,
            "generated_text": final_generated_text,
            "step_analyses": generation_results,
        }

    def analyze_step(
            self,
            logits: torch.Tensor,
            attentions: List[torch.Tensor],
            hidden_states: List[torch.Tensor],
    ) -> Dict:
        """Analyze a single generation step."""
        step_analysis = {}

        if self.config.logits_entropy.enabled:
            probs = F.softmax(logits[:, -1, :], dim=-1)
            logits_entropy = self.calculate_entropy(probs[0].cpu().numpy())
            step_analysis["logits_entropy"] = logits_entropy

        if self.config.attention_entropy.enabled:
            attention_entropies = self.calculate_attention_entropy(attentions)
            # For simplicity, take the mean entropy across all heads
            mean_attn_entropy = np.mean(attention_entropies)
            step_analysis["attention_entropy"] = mean_attn_entropy

        if self.config.mc_dropout.enabled:
            mean_probs, var_probs = self.mc_dropout(
                {"input_ids": logits.argmax(dim=-1)}, n_samples=10
            )
            uncertainty = var_probs[0, -1, :].mean().item()
            step_analysis["mc_dropout_uncertainty"] = uncertainty

        if self.config.perplexity.enabled:
            # Perplexity is typically calculated on complete sentences, so it's skipped here
            perplexity = self.calculate_perplexity(
                self.tokenizer.decode(logits.argmax(dim=-1)[0], skip_special_tokens=True)
            )
            step_analysis["perplexity"] = perplexity

        # Categorize state for the current step
        if (
                "logits_entropy" in step_analysis
                and "attention_entropy" in step_analysis
        ):
            logits_entropy = step_analysis["logits_entropy"]
            attention_entropy = step_analysis["attention_entropy"]
            if self.calibrated:
                state = self.categorize_state_ml(logits_entropy, attention_entropy)
            else:
                state = self.categorize_state(logits_entropy, attention_entropy)
            step_analysis["model_state"] = state

        return step_analysis

    def sample_next_token(
            self, logits: torch.Tensor, method: str, **kwargs
    ) -> torch.Tensor:
        """Sample the next token based on the specified method."""
        if method == "temperature":
            temperature = kwargs.get("temperature", 1.0)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        elif method == "top-k":
            k = kwargs.get("k", 50)
            logits = logits[:, -1, :]
            filtered_logits = self.top_k_top_p_filtering(logits, top_k=k)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        elif method == "top-p":
            p = kwargs.get("p", 0.9)
            logits = logits[:, -1, :]
            filtered_logits = self.top_k_top_p_filtering(logits, top_p=p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return next_token_id

    def visualize_attention(
            self, attentions: List[torch.Tensor], input_tokens: List[str]
    ) -> None:
        """Visualize attention heatmap for the last layer."""
        if not self.config.attention_entropy.enabled:
            print("Attention visualization is not enabled in the configuration.")
            return

        try:
            last_layer_attn = attentions[-1][0]  # [num_heads, seq_len, seq_len]
            avg_attn = last_layer_attn.mean(dim=0)  # [seq_len, seq_len]

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                avg_attn.cpu().numpy(),
                xticklabels=input_tokens,
                yticklabels=input_tokens,
                cmap="viridis",
            )
            plt.title("Attention Heatmap (Last Layer)")
            plt.xlabel("Key Positions")
            plt.ylabel("Query Positions")
            plt.show()
        except Exception as e:
            print(f"Error in attention visualization: {e}")

    def visualize_entropy_over_time(self, generation_results: Dict):
        """Visualize logits and attention entropy over generation steps."""
        if not (self.config.logits_entropy.enabled and self.config.attention_entropy.enabled):
            print(
                "Logits and attention entropy visualization is not enabled in the configuration."
            )
            return

        steps = range(1, len(generation_results["step_analyses"]) + 1)
        logits_entropies = [
            step["logits_entropy"] for step in generation_results["step_analyses"]
        ]
        attention_entropies = [
            step["attention_entropy"] for step in generation_results["step_analyses"]
        ]

        plt.figure(figsize=(12, 6))
        plt.plot(
            steps, logits_entropies, label="Logits Entropy", marker="o", color="blue"
        )
        plt.plot(
            steps,
            attention_entropies,
            label="Attention Entropy",
            marker="s",
            color="orange",
        )
        plt.xlabel("Generation Step")
        plt.ylabel("Entropy")
        plt.title("Entropy Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_model_states(self, generation_results: Dict):
        """Visualize model states over generation steps."""
        if not any(
                "model_state" in step for step in generation_results["step_analyses"]
        ):
            print("Model state visualization is not available.")
            return

        steps = range(1, len(generation_results["step_analyses"]) + 1)
        states = [
            step["model_state"] for step in generation_results["step_analyses"]
            if "model_state" in step
        ]

        state_to_num = {"Uncertain": 0, "Confident": 1, "Overconfident": 2}
        numeric_states = [state_to_num[state] for state in states]

        plt.figure(figsize=(12, 6))
        plt.plot(steps, numeric_states, marker="o", linestyle="-")
        plt.yticks([0, 1, 2], ["Uncertain", "Confident", "Overconfident"])
        plt.xlabel("Generation Step")
        plt.ylabel("Model State")
        plt.title("Model State Over Time")
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


# Example usage
if __name__ == "__main__":
    config = EntropyAnalysisConfig()

    wrapper = BasicEntropyAnalysisWrapper("meta-llama/Llama-3.2-1B-Instruct", config=config)

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
        method="temperature",
        temperature=0.3,
    )

    # Visualize results
    wrapper.visualize_entropy_over_time(generation_results)
    wrapper.visualize_model_states(generation_results)
