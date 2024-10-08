import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList, LogitsProcessor, GenerationConfig
from typing import List, Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass

LN_2 = 0.69314718056  # ln(2)


@dataclass
class SamplerConfig:
    temp: float = 0.666
    top_p: float = 0.90
    top_k: int = 27
    min_p: float = 0.03
    low_ent_thresh: float = 0.1
    low_vent_thresh: float = 0.1
    med_ent_thresh: float = 3.0
    high_ent_thresh: float = 5.0
    high_vent_thresh: float = 5.0
    n_adaptive_samples: int = 5
    helv_attn_ent_offset: float = 1.3
    helv_attn_ent_coef: float = 0.2
    lehv_interaction_strength_offset: float = 1.2
    lehv_interaction_strength_coef: float = 0.3
    hehv_attn_ent_coef: float = 0.2
    hehv_attn_vent_offset: float = 2.0
    hehv_attn_vent_coef: float = 0.5
    ada_temp_logits: float = 0.3
    ada_temp_attn: float = 0.2
    ada_temp_agree: float = 0.2
    ada_top_p: float = 0.1
    ada_top_k_int: float = 0.3
    ada_top_k_agree: float = 0.2
    ada_min_p: float = 0.5
    ada_score_logits_ent: float = 0.1
    ada_score_attn_ent: float = 0.2
    ada_score_logits_vent: float = 0.3
    ada_score_attn_vent: float = 0.4
    ada_score_agree: float = 0.5
    ada_score_int: float = 0.6


class HookBasedEntropySampler(LogitsProcessor):
    def __init__(
            self,
            model,
            tokenizer,
            pause_token_id: int,
            clarifying_question_token_id: int,
            config: Optional[SamplerConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pause_token_id = pause_token_id
        self.clarifying_question_token_id = clarifying_question_token_id
        self.config = config or SamplerConfig()
        self.last_attention_scores = torch.Tensor
        self._register_hooks()

    def attention_hook(self, module, input, output):
        #print(f"Attention hook called. Output type: {type(output)}")
        if isinstance(output, tuple) and output[0] is not None:
            self.last_attention_scores = output[0].detach()
            #print(f"Attention scores shape in hook: {self.last_attention_scores.shape}")
        #else:
        #print("Attention scores not found in expected format")

    def _register_hooks(self):
        if hasattr(self.model, 'model'):
            for layer in self.model.model.layers:
                layer.self_attn.register_forward_hook(self.attention_hook)
        else:
            raise ValueError("Unsupported model architecture. Expected 'model.model.layers'.")

    def calculate_varentropy_logsoftmax(self, logits: torch.Tensor, axis: int = -1) -> Tuple[
        torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(logits, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
        varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1)) ** 2, dim=axis)
        return entropy, varentropy

    def calculate_metrics(self, logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)

        #print(f"Raw entropy: {entropy}, Raw varentropy: {varentropy}")

        # Replace inf and nan values
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=10.0, neginf=0.0)
        varentropy = torch.nan_to_num(varentropy, nan=0.0, posinf=10.0, neginf=0.0)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
        attn_varentropy = torch.var(attn_entropy, dim=-1)
        mean_attention = torch.mean(attention_probs, dim=-2)
        agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(-2)), dim=(-2, -1))
        interaction_strength = torch.mean(torch.abs(attention_scores))

        return {
            "logits_entropy": torch.mean(entropy),
            "logits_varentropy": torch.mean(varentropy),
            "attn_entropy": torch.mean(attn_entropy),
            "attn_varentropy": torch.mean(attn_varentropy),
            "agreement": torch.mean(agreement),
            "interaction_strength": interaction_strength
        }

    def adaptive_sample(
            self,
            logits: torch.Tensor,
            metrics: Dict[str, torch.Tensor],
            input_ids: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

        # Ensure all values are finite
        logits_uncertainty = torch.nan_to_num(logits_uncertainty, nan=0.0, posinf=10.0, neginf=0.0)
        attn_uncertainty = torch.nan_to_num(attn_uncertainty, nan=0.0, posinf=10.0, neginf=0.0)

        # Clamp values to prevent extreme adjustments
        temperature = torch.clamp(cfg.temp * (1 + cfg.ada_temp_logits * logits_uncertainty +
                                              cfg.ada_temp_attn * attn_uncertainty -
                                              cfg.ada_temp_agree * metrics["agreement"]), min=0.1, max=2.0)
        top_p = torch.clamp(cfg.top_p * (1 + cfg.ada_top_p * metrics["attn_varentropy"]), 0.1, 1.0)
        top_k = int(torch.clamp(
            torch.round(cfg.top_k * (1 + cfg.ada_top_k_int * metrics["interaction_strength"] -
                                     cfg.ada_top_k_agree * metrics["agreement"])),
            min=1,
            max=100
        ).item())
        min_p = torch.clamp(cfg.min_p * (1 - cfg.ada_min_p * logits_uncertainty), 0.01, 0.5)

        #print(f"Adaptive sampling parameters:")
        #print(f"Temperature: {temperature}")
        #print(f"Top-p: {top_p}")
        #print(f"Top-k: {top_k}")
        #print(f"Min-p: {min_p}")

        samples = []
        for _ in range(cfg.n_adaptive_samples):
            sample = self._sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
            samples.append(sample)

        sample_scores = torch.stack([self._score_sample(sample, logits, metrics) for sample in samples])
        best_sample_idx = torch.argmax(sample_scores)

        #print(f"Selected sample index: {best_sample_idx}")
        #print(f"Selected sample score: {sample_scores[best_sample_idx]}")

        return samples[best_sample_idx]

    def _sample(
            self,
            logits: torch.Tensor,
            temperature: float,
            top_p: float,
            top_k: int,
            min_p: float
    ) -> torch.Tensor:
        # Apply temperature
        logits = logits / temperature

        # Remove inf and -inf values
        logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)

        # Sort logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        # Calculate cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold (top_p)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., :top_k] = 0

        # Remove tokens below min_p threshold
        min_p_mask = sorted_probs < min_p
        sorted_indices_to_remove = sorted_indices_to_remove | min_p_mask

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        # Set logits to -inf where indices_to_remove is True
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Ensure at least one token can be chosen
        if (logits == float('-inf')).all():
            logits = torch.zeros_like(logits)
            logits[..., 0] = 1.0

        # Calculate probabilities
        probs = F.softmax(logits, dim=-1)

        # Ensure no NaN or inf values in probs
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # Renormalize if necessary
        if not torch.allclose(probs.sum(), torch.tensor(1.0)):
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    def _score_sample(
            self,
            sample: torch.Tensor,
            logits: torch.Tensor,
            metrics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        cfg = self.config
        log_prob = torch.sum(F.log_softmax(logits, dim=-1) * F.one_hot(sample, logits.shape[-1]))
        confidence_score = (
                (1 - metrics["logits_entropy"]) * cfg.ada_score_logits_ent +
                (1 - metrics["attn_entropy"]) * cfg.ada_score_attn_ent +
                (1 - metrics["logits_varentropy"]) * cfg.ada_score_logits_vent +
                (1 - metrics["attn_varentropy"]) * cfg.ada_score_attn_vent +
                metrics["agreement"] * cfg.ada_score_agree +
                metrics["interaction_strength"] * cfg.ada_score_int
        )
        return log_prob + confidence_score

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.last_attention_scores is None:
            raise ValueError("No attention scores available. Make sure the model forward pass has been called.")

        #print(f"Scores shape: {scores.shape}")
        #print(f"Scores min: {scores.min()}, max: {scores.max()}, mean: {scores.mean()}")
        #print(f"Scores contain inf: {torch.isinf(scores).any()}, contain nan: {torch.isnan(scores).any()}")

        metrics = self.calculate_metrics(scores, self.last_attention_scores)
        ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
        cfg = self.config

        #print(f"Calculated Entropy: {ent}, Varentropy: {vent}")

        if ent < cfg.low_ent_thresh and vent < cfg.low_vent_thresh:
            #print("Low entropy and varentropy: using argmax")
            return torch.argmax(scores, dim=-1, keepdim=True)
        elif ent > cfg.high_ent_thresh and vent < cfg.low_vent_thresh:
            if not torch.isin(input_ids[:, -1], torch.tensor([self.clarifying_question_token_id])).any():
                #print("High entropy, low varentropy: using clarifying question")
                return torch.full_like(input_ids[:, -1:], self.clarifying_question_token_id)
            else:
                #print("High entropy, low varentropy: adjusting temperature")
                temp_adj = cfg.helv_attn_ent_offset + cfg.helv_attn_ent_coef * metrics["attn_entropy"]
                return self._sample(scores, temperature=min(1.5, cfg.temp * temp_adj),
                                    top_p=cfg.top_p, top_k=cfg.top_k, min_p=cfg.min_p)
        elif ent < cfg.high_ent_thresh and vent > cfg.high_vent_thresh:
            #print("Low entropy, high varentropy: adjusting temperature and top_k")
            temp_adj = cfg.lehv_interaction_strength_offset + cfg.lehv_interaction_strength_coef * metrics[
                "interaction_strength"]
            top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - metrics["agreement"]))))
            return self._sample(scores, temperature=min(1.5, cfg.temp * temp_adj),
                                top_p=cfg.top_p, top_k=top_k_adj, min_p=cfg.min_p)
        elif ent > cfg.med_ent_thresh and vent > cfg.high_vent_thresh:
            #print("High entropy and varentropy: adjusting temperature and top_p")
            temp_adj = cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * metrics["attn_varentropy"]
            top_p_adj = max(0.5, cfg.top_p - cfg.hehv_attn_ent_coef * metrics["attn_entropy"])
            return self._sample(scores, temperature=max(2.0, cfg.temp * temp_adj),
                                top_p=top_p_adj, top_k=cfg.top_k, min_p=cfg.min_p)
        else:
            #print("Using adaptive sampling")
            return self.adaptive_sample(scores, metrics, input_ids)


# Example usage
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


# Assuming "..." is our pause token and "?" is our clarifying question token
pause_token_id = tokenizer.convert_tokens_to_ids("...")
clarifying_question_token_id = tokenizer.convert_tokens_to_ids("?")

entropy_sampler = HookBasedEntropySampler(model, tokenizer, pause_token_id, clarifying_question_token_id)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the concept of entropy in thermodynamics."},
]
# Tokenize the chat and create attention mask
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
attention_mask = tokenized_chat.ne(tokenizer.eos_token_id).long()

# Set up the generation configuration
generation_config = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.3,
    top_p=1.0,
    num_return_sequences=1,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
)

# Generate output with attention mask
output_ids = model.generate(
    tokenized_chat,
    attention_mask=attention_mask,  # Pass the attention mask
    generation_config=generation_config,
    logits_processor=[entropy_sampler]
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
