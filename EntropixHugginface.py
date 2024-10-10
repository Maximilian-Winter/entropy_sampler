import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


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


class EntropixHuggingFace:
    def __init__(self, model_name: str, sampler_config: Optional[SamplerConfig] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.config = sampler_config or SamplerConfig()

    def generate(self, prompt: str, max_length: int = 100) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        past_key_values = None
        generated_tokens = []

        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            attention_scores = outputs.attentions[-1]  # Use the last layer's attention

            next_token = self.sample_token(input_ids, logits, attention_scores)
            generated_tokens.append(next_token.item())

            input_ids = next_token.unsqueeze(0)
            attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_tokens)

    def sample_token(self, gen_tokens: torch.Tensor, logits: torch.Tensor,
                     attention_scores: torch.Tensor) -> torch.Tensor:
        metrics = self.calculate_metrics(logits, attention_scores)
        ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]

        if ent < self.config.low_ent_thresh and vent < self.config.low_vent_thresh:
            # Low Entropy, Low Varentropy: "flowing with unspoken intent"
            return torch.argmax(logits, dim=-1)

        elif ent > self.config.high_ent_thresh and vent < self.config.low_vent_thresh:
            # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
            temp_adj = 1.3 + 0.2 * metrics["attn_entropy"]
            return self._sample(logits, temperature=min(1.5, self.config.temp * temp_adj))

        elif ent < self.config.high_ent_thresh and vent > self.config.high_vent_thresh:
            # Low Entropy, High Varentropy: "exploring forks in the path"
            temp_adj = 1.2 + 0.3 * metrics["interaction_strength"]
            top_k_adj = max(5, int(self.config.top_k * (1 + 0.5 * (1 - metrics["agreement"]))))
            return self._sample(logits, temperature=min(1.5, self.config.temp * temp_adj), top_k=top_k_adj)

        elif ent > self.config.med_ent_thresh and vent > self.config.high_vent_thresh:
            # High Entropy, High Varentropy: "resampling in the mist"
            temp_adj = 2.0 + 0.5 * metrics["attn_varentropy"]
            top_p_adj = max(0.5, self.config.top_p - 0.2 * metrics["attn_entropy"])
            return self._sample(logits, temperature=max(2.0, self.config.temp * temp_adj), top_p=top_p_adj)

        else:
            # Middle ground: use adaptive sampling
            return self.adaptive_sample(logits, metrics)

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None,
                top_p: Optional[float] = None) -> torch.Tensor:
        logits = logits / temperature

        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)

        if top_p is not None:
            logits = self.top_p_logits(logits, top_p)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def adaptive_sample(self, logits: torch.Tensor, metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

        temperature = self.config.temp * (
                    1 + 0.3 * logits_uncertainty + 0.2 * attn_uncertainty - 0.2 * metrics["agreement"])
        top_p = min(1.0, max(0.1, self.config.top_p * (1 + 0.1 * metrics["attn_varentropy"])))
        top_k = max(1, min(100, int(self.config.top_k * (
                    1 + 0.3 * metrics["interaction_strength"] - 0.2 * metrics["agreement"]))))
        min_p = max(0.01, min(0.5, self.config.min_p * (1 - 0.5 * logits_uncertainty)))

        logits = logits / temperature
        logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, min_p=min_p)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
        v, _ = torch.topk(logits, k)
        return torch.where(logits < v[:, [-1]], torch.full_like(logits, -float('Inf')), logits)

    @staticmethod
    def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        return torch.where(indices_to_remove, torch.full_like(logits, -float('Inf')), logits)

    @staticmethod
    def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0,
                              filter_value: float = -float('Inf')) -> torch.Tensor:
        if top_k > 0:
            logits = EntropixHuggingFace.top_k_logits(logits, top_k)
        if top_p < 1.0:
            logits = EntropixHuggingFace.top_p_logits(logits, top_p)
        if min_p > 0.0:
            probs = F.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1, keepdim=True).values
            logits[probs < min_p * max_probs] = filter_value
        return logits

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

    @staticmethod
    def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(logits, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / torch.log(torch.tensor(2.0))
        varentropy = torch.sum(probs * (log_probs / torch.log(torch.tensor(2.0)) + entropy.unsqueeze(-1)) ** 2,
                               dim=axis)
        return entropy, varentropy
