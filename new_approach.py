import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from transformers import LogitsProcessor
import math

LN_2 = 0.69314718056  # ln(2)

class SamplerConfig:
    def __init__(self):
        self.temp = 0.666
        self.top_p = 0.90
        self.top_k = 27
        self.min_p = 0.03
        self.low_ent_thresh = 0.1
        self.low_vent_thresh = 0.1
        self.med_ent_thresh = 3.0
        self.high_ent_thresh = 5.0
        self.high_vent_thresh = 5.0
        self.helv_attn_ent_offset = 1.3
        self.helv_attn_ent_coef = 0.2
        self.lehv_interaction_strength_offset = 1.2
        self.lehv_interaction_strength_coef = 0.3
        self.hehv_attn_ent_coef = 0.2
        self.hehv_attn_vent_offset = 2.0
        self.hehv_attn_vent_coef = 0.5
        self.n_adaptive_samples = 5
        self.ada_temp_logits = 0.3
        self.ada_temp_attn = 0.2
        self.ada_temp_agree = 0.2
        self.ada_top_p = 0.1
        self.ada_top_k_int = 0.3
        self.ada_top_k_agree = 0.2
        self.ada_min_p = 0.5
        self.ada_score_logits_ent = 0.1
        self.ada_score_attn_ent = 0.2
        self.ada_score_logits_vent = 0.3
        self.ada_score_attn_vent = 0.4
        self.ada_score_agree = 0.5
        self.ada_score_int = 0.6

class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self, config: SamplerConfig, clarifying_question_token: int = 2564):
        self.config = config
        self.clarifying_question_token = clarifying_question_token
        self.attention_maps = None

    def set_attention_maps(self, attention_maps: torch.Tensor):
        self.attention_maps = attention_maps

    def calculate_varentropy_logsoftmax(self, logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(logits, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2
        varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
        return entropy, varentropy

    def calculate_metrics(self, logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)

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

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int, min_p: float) -> torch.Tensor:
        logits = logits[:, -1, :]
        probs = F.softmax(logits / temperature, dim=-1)

        # Apply min_p sampling
        if min_p > 0.0:
            p_max = torch.max(probs, dim=-1, keepdim=True)[0]
            indices_to_remove = probs < (min_p * p_max)
            logits = torch.where(indices_to_remove, torch.full_like(logits, float('-inf')), logits)

        # Apply top-k sampling
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        probs_sort = torch.flip(top_k_probs, dims=[-1])
        probs_idx = torch.flip(top_k_indices, dims=[-1])
        probs_sum = torch.cumsum(probs_sort, dim=-1)

        # Apply top-p sampling
        mask = torch.where(probs_sum - probs_sort > top_p, torch.ones_like(probs_sort), torch.zeros_like(probs_sort))
        probs_sort = probs_sort * (1 - mask)
        probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)

        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, dim=-1, index=next_token)

        return next_token.int()

    def score_sample(self, sample: torch.Tensor, logits: torch.Tensor, metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        log_prob = torch.sum(F.log_softmax(logits, dim=-1) * F.one_hot(sample, logits.shape[-1]))
        confidence_score = (
                (1 - metrics["logits_entropy"]) * self.config.ada_score_logits_ent +
                (1 - metrics["attn_entropy"]) * self.config.ada_score_attn_ent +
                (1 - metrics["logits_varentropy"]) * self.config.ada_score_logits_vent +
                (1 - metrics["attn_varentropy"]) * self.config.ada_score_attn_vent +
                metrics["agreement"] * self.config.ada_score_agree +
                metrics["interaction_strength"] * self.config.ada_score_int
        )
        return log_prob + confidence_score

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.attention_maps is None:
            raise ValueError("Attention maps have not been set. Call set_attention_maps() before processing.")

        metrics = self.calculate_metrics(scores, self.attention_maps)
        ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
        attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
        agreement = metrics["agreement"]
        interaction_strength = metrics["interaction_strength"]

        # Low Entropy, Low Varentropy
        if ent < self.config.low_ent_thresh and vent < self.config.low_vent_thresh:
            return torch.argmax(scores[:, -1], dim=-1, keepdim=True).int()

        # High Entropy, Low Varentropy
        elif ent > self.config.high_ent_thresh and vent < self.config.low_vent_thresh:
            if not torch.isin(input_ids[:, -1], self.clarifying_question_token).any():
                return torch.full((scores.shape[0], 1), self.clarifying_question_token, dtype=torch.int64)
            else:
                temp_adj = self.config.helv_attn_ent_offset + self.config.helv_attn_ent_coef * attn_ent
                return self._sample(scores, temperature=min(1.5, self.config.temp * temp_adj),
                                    top_p=self.config.top_p, top_k=self.config.top_k, min_p=self.config.min_p)

        # Low Entropy, High Varentropy
        elif ent < self.config.high_ent_thresh and vent > self.config.high_vent_thresh:
            temp_adj = self.config.lehv_interaction_strength_offset + self.config.lehv_interaction_strength_coef * interaction_strength
            top_k_adj = max(5, int(self.config.top_k * (1 + 0.5 * (1 - agreement))))
            return self._sample(scores, temperature=min(1.5, self.config.temp * temp_adj),
                                top_p=self.config.top_p, top_k=top_k_adj, min_p=self.config.min_p)

        # High Entropy, High Varentropy
        elif ent > self.config.med_ent_thresh and vent > self.config.high_vent_thresh:
            temp_adj = self.config.hehv_attn_vent_offset + self.config.hehv_attn_vent_coef * attn_vent
            top_p_adj = max(0.5, self.config.top_p - self.config.hehv_attn_ent_coef * attn_ent)
            return self._sample(scores, temperature=max(2.0, self.config.temp * temp_adj),
                                top_p=top_p_adj, top_k=self.config.top_k, min_p=self.config.min_p)

        # Middle ground: use adaptive sampling
        else:
            logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
            attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

            temperature = self.config.temp * (1 + self.config.ada_temp_logits * logits_uncertainty +
                                              self.config.ada_temp_attn * attn_uncertainty -
                                              self.config.ada_temp_agree * metrics["agreement"])
            top_p = torch.clamp(self.config.top_p * (1 + self.config.ada_top_p * metrics["attn_varentropy"]), 0.1, 1.0)
            top_k = int(torch.clamp(
                torch.round(self.config.top_k * (1 + self.config.ada_top_k_int * metrics["interaction_strength"] -
                                                 self.config.ada_top_k_agree * metrics["agreement"])),
                min=1, max=100
            ))
            min_p = torch.clamp(self.config.min_p * (1 - self.config.ada_min_p * logits_uncertainty), 0.01, 0.5)

            samples = []
            for _ in range(self.config.n_adaptive_samples):
                sample = self._sample(scores, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
                samples.append(sample)

            sample_scores = torch.stack([self.score_sample(sample, scores, metrics) for sample in samples])
            best_sample_idx = torch.argmax(sample_scores)
            return samples[best_sample_idx]

# Usage example:
config = SamplerConfig()
logits_processor = CustomLogitsProcessor(config)

# In your generation loop:
def get_attention_maps_hook(module, input, output):
    logits_processor.set_attention_maps(output.attentions[-1])  # Use the last layer's attention

# Attach the hook to the model
model.base_model.encoder.layer[-1].attention.register_forward_hook(get_attention_maps_hook)

# Generate tokens
generated = model.generate(
    input_ids,
    max_length=max_length,
    logits_processor=[logits_processor],
    output_attentions=True,
    return_dict_in_generate=True
)