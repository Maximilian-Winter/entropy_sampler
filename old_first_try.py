import math
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import GenerationConfig

LN_2 = 0.69314718056  # ln(2)


@dataclass
class EntropyBasedSamplerConfig(GenerationConfig):
    """Configuration class for EntropyBasedSampler."""
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
    clarifying_question_token: int = 2564


class EntropyBasedSampler(LogitsProcessor):
    """Entropy-based sampler for language models."""

    def __init__(self, config: EntropyBasedSamplerConfig):
        self.config = config
        self.last_attention_scores = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.last_attention_scores is None:
            raise ValueError("Attention scores not available. Make sure to register the hook.")

        metrics = self._calculate_metrics(scores, self.last_attention_scores)
        return self._sample_based_on_metrics(input_ids, scores, metrics)

    def _calculate_metrics(self, logits: torch.FloatTensor, attention_scores: torch.FloatTensor) -> Dict[
        str, torch.FloatTensor]:
        entropy, varentropy = self._calculate_varentropy_logsoftmax(logits)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
        attn_varentropy = torch.var(attn_entropy, dim=1)
        mean_attention = torch.mean(attention_probs, dim=1)
        agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))
        interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))

        return {
            "logits_entropy": torch.mean(entropy),
            "logits_varentropy": torch.mean(varentropy),
            "attn_entropy": torch.mean(attn_entropy),
            "attn_varentropy": torch.mean(attn_varentropy),
            "agreement": torch.mean(agreement),
            "interaction_strength": interaction_strength
        }

    def _calculate_varentropy_logsoftmax(self, logits: torch.FloatTensor) -> Tuple[
        torch.FloatTensor, torch.FloatTensor]:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1) / LN_2
        varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1)) ** 2, dim=-1)
        return entropy, varentropy

    def _sample_based_on_metrics(self, gen_tokens: torch.LongTensor, logits: torch.FloatTensor,
                                 metrics: Dict[str, torch.FloatTensor]) -> torch.FloatTensor:
        ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
        attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
        agreement = metrics["agreement"]
        interaction_strength = metrics["interaction_strength"]

        if ent < self.config.low_ent_thresh and vent < self.config.low_vent_thresh:
            return F.one_hot(torch.argmax(logits[:, -1], dim=-1), num_classes=logits.size(-1)).float()
        elif ent > self.config.high_ent_thresh and vent < self.config.low_vent_thresh:
            if not torch.isin(gen_tokens[:, -1], torch.tensor([self.config.clarifying_question_token])).any():
                return F.one_hot(torch.tensor([self.config.clarifying_question_token]),
                                 num_classes=logits.size(-1)).float()
            else:
                temp_adj = self.config.helv_attn_ent_offset + self.config.helv_attn_ent_coef * attn_ent
                return self._sample(logits, temperature=min(1.5, self.config.temp * temp_adj))
        elif ent < self.config.high_ent_thresh and vent > self.config.high_vent_thresh:
            temp_adj = self.config.lehv_interaction_strength_offset + self.config.lehv_interaction_strength_coef * interaction_strength
            top_k_adj = max(5, int(self.config.top_k * (1 + 0.5 * (1 - agreement))))
            return self._sample(logits, temperature=min(1.5, self.config.temp * temp_adj), top_k=top_k_adj)
        elif ent > self.config.med_ent_thresh and vent > self.config.high_vent_thresh:
            temp_adj = self.config.hehv_attn_vent_offset + self.config.hehv_attn_vent_coef * attn_vent
            top_p_adj = max(0.5, self.config.top_p - self.config.hehv_attn_ent_coef * attn_ent)
            return self._sample(logits, temperature=max(2.0, self.config.temp * temp_adj), top_p=top_p_adj)
        else:
            return self._adaptive_sample(logits, metrics)

    def _sample(self, logits: torch.FloatTensor, temperature: float = 1.0, top_k: Optional[int] = None,
                top_p: Optional[float] = None, min_p: Optional[float] = None) -> torch.FloatTensor:
        logits = logits / temperature

        if min_p is not None:
            p_max = torch.max(F.softmax(logits, dim=-1), dim=-1, keepdim=True).values
            logits = torch.where(F.softmax(logits, dim=-1) < (min_p * p_max), torch.full_like(logits, float('-inf')),
                                 logits)

        if top_k is not None:
            top_k_logits, _ = torch.topk(logits, k=top_k)
            logits = torch.where(logits < top_k_logits[..., -1:], torch.full_like(logits, float('-inf')), logits)

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        return F.softmax(logits, dim=-1)

    def _adaptive_sample(self, logits: torch.FloatTensor, metrics: Dict[str, torch.FloatTensor]) -> torch.FloatTensor:
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

        temperature = self.config.temp * (
                    1 + self.config.ada_temp_logits * logits_uncertainty + self.config.ada_temp_attn * attn_uncertainty - self.config.ada_temp_agree *
                    metrics["agreement"])
        top_p = torch.clamp(self.config.top_p * (1 + self.config.ada_top_p * metrics["attn_varentropy"]), 0.1, 1.0)
        top_k = int(torch.clamp(
            torch.round(torch.tensor(self.config.top_k) * (
                        1 + self.config.ada_top_k_int * metrics["interaction_strength"] - self.config.ada_top_k_agree *
                        metrics["agreement"])),
            min=1,
            max=100
        ).item())
        min_p = torch.clamp(self.config.min_p * (1 - self.config.ada_min_p * logits_uncertainty), 0.01, 0.5)

        samples = []
        for _ in range(self.config.n_adaptive_samples):
            sample = self._sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
            samples.append(sample)

        sample_scores = torch.tensor([self._score_sample(logits, s, metrics) for s in samples])
        return samples[torch.argmax(sample_scores)]

    def _score_sample(self, logits: torch.FloatTensor, sample: torch.FloatTensor,
                      metrics: Dict[str, torch.FloatTensor]) -> float:
        log_prob = torch.sum(torch.log(sample) * F.softmax(logits, dim=-1))
        confidence_score = (
                (1 - metrics["logits_entropy"]) * self.config.ada_score_logits_ent +
                (1 - metrics["attn_entropy"]) * self.config.ada_score_attn_ent +
                (1 - metrics["logits_varentropy"]) * self.config.ada_score_logits_vent +
                (1 - metrics["attn_varentropy"]) * self.config.ada_score_attn_vent +
                metrics["agreement"] * self.config.ada_score_agree +
                metrics["interaction_strength"] * self.config.ada_score_int
        )
        return log_prob + confidence_score


def create_entropy_based_sampler(config: Optional[EntropyBasedSamplerConfig] = None) -> EntropyBasedSampler:
    """Create an instance of EntropyBasedSampler with the given configuration."""
    if config is None:
        config = EntropyBasedSamplerConfig()
    return EntropyBasedSampler(config)


def register_attention_hook(model: PreTrainedModel, sampler: EntropyBasedSampler):
    """
    Register a hook to capture attention scores from the model.

    Args:
        model (PreTrainedModel): The Hugging Face transformer model.
        sampler (EntropyBasedSampler): The entropy-based sampler instance.
    """

    def hook_fn(module, input, output):
        sampler.last_attention_scores = output[2][-1]  # Assuming the last layer's attention scores

    # Find the last attention layer and register the hook
    for name, module in reversed(list(model.named_modules())):
        if "attention" in name.lower():
            module.register_forward_hook(hook_fn)
            break

# Usage example:
model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")

sampler_config = EntropyBasedSamplerConfig(temp=0.7, top_p=0.9, top_k=50)
entropy_sampler = create_entropy_based_sampler(sampler_config)

register_attention_hook(model, entropy_sampler)

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    logits_processor=[entropy_sampler],
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
