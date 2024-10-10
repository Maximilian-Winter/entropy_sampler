import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class EntropyAwareModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 config: Optional[SamplerConfig] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config or SamplerConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model.to(self.device)
        self.to(self.device)

        # Initialize EMA for entropy and varentropy
        self.ema_entropy = 0
        self.ema_varentropy = 0
        self.ema_alpha = 0.1  # Adjust this value to control the EMA update rate

        logger.info(f"Initialized EntropyAwareModel using device: {self.device}")

    def calculate_varentropy_logsoftmax(self, logits: torch.Tensor, axis: int = -1) -> Tuple[
        torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(logits, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / math.log(2)  # Convert to base-2
        varentropy = torch.sum(probs * (log_probs / math.log(2) + entropy.unsqueeze(-1)) ** 2, dim=axis)
        return entropy, varentropy

    def calculate_metrics(self, logits: torch.Tensor, attention_scores: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)

        attn_scores = torch.stack(attention_scores, dim=0)
        current_token_attn = attn_scores[:, :, :, -1, :]  # [num_layers, batch_size, num_heads, seq_len]

        attention_probs = F.softmax(current_token_attn, dim=-1)
        attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
        attn_varentropy = torch.var(attn_entropy, dim=-1)

        mean_attention = torch.mean(attention_probs, dim=1)
        agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))

        interaction_strength = torch.mean(torch.abs(current_token_attn), dim=(1, 2, 3))

        return {
            "logits_entropy": torch.mean(entropy),
            "logits_varentropy": torch.mean(varentropy),
            "attn_entropy": torch.mean(attn_entropy),
            "attn_varentropy": torch.mean(attn_varentropy),
            "agreement": torch.mean(agreement),
            "interaction_strength": torch.mean(interaction_strength)
        }

    def _sample(self, logits: torch.FloatTensor, temperature: float, top_p: float, top_k: int,
                min_p: float) -> torch.LongTensor:
        logits = logits / temperature
        logits = self.adjust_scores(logits, top_p, top_k, min_p)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def adjust_scores(self, scores: torch.FloatTensor, top_p: float, top_k: int, min_p: float) -> torch.FloatTensor:
        top_k = min(top_k, scores.size(-1))
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores[indices_to_remove] = float('-inf')

        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        scores[indices_to_remove] = float('-inf')

        scores[F.softmax(scores, dim=-1) < min_p] = float('-inf')
        return scores

    def adaptive_sample(self, logits: torch.FloatTensor, metrics: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        cfg = self.config
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

        temperature = cfg.temp * (
                    1 + cfg.ada_temp_logits * logits_uncertainty + cfg.ada_temp_attn * attn_uncertainty - cfg.ada_temp_agree *
                    metrics["agreement"])
        top_p = torch.clamp(cfg.top_p * (1 + cfg.ada_top_p * metrics["attn_varentropy"]), 0.1, 1.0)
        top_k = int(torch.clamp(
            torch.round(cfg.top_k * (
                        1 + cfg.ada_top_k_int * metrics["interaction_strength"] - cfg.ada_top_k_agree * metrics[
                    "agreement"])),
            min=1,
            max=100
        ).item())
        min_p = torch.clamp(cfg.min_p * (1 - cfg.ada_min_p * logits_uncertainty), 0.01, 0.5)

        samples = []
        for _ in range(cfg.n_adaptive_samples):
            sample = self._sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
            samples.append(sample)

        sample_scores = [self._score_sample(sample, logits, metrics) for sample in samples]
        best_sample_idx = torch.argmax(torch.tensor(sample_scores))
        return samples[best_sample_idx]

    def _score_sample(self, sample: torch.Tensor, logits: torch.Tensor, metrics: Dict[str, torch.Tensor]) -> float:
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

    def update_ema(self, entropy: float, varentropy: float):
        self.ema_entropy = self.ema_alpha * entropy + (1 - self.ema_alpha) * self.ema_entropy
        self.ema_varentropy = self.ema_alpha * varentropy + (1 - self.ema_alpha) * self.ema_varentropy

    def entropy_based_sampling(self, gen_tokens: torch.LongTensor, logits: torch.FloatTensor,
                               metrics: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
        cfg = self.config

        # Low Entropy, Low Varentropy: "flowing with unspoken intent"
        if ent < cfg.low_ent_thresh and vent < cfg.low_vent_thresh:
            return torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)

        # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
        elif ent > cfg.high_ent_thresh and vent < cfg.low_vent_thresh:
            clarifying_question_token_id = self.tokenizer.convert_tokens_to_ids("?")
            if not torch.isin(gen_tokens[:, -1],
                              torch.tensor([clarifying_question_token_id], device=self.device)).any():
                return torch.tensor([[clarifying_question_token_id]], dtype=torch.int32, device=self.device)
            else:
                temp_adj = cfg.helv_attn_ent_offset + cfg.helv_attn_ent_coef * metrics["attn_entropy"]
                return self._sample(logits, temperature=min(1.5, cfg.temp * temp_adj.item()), top_p=cfg.top_p,
                                    top_k=cfg.top_k, min_p=cfg.min_p)

        # Low Entropy, High Varentropy: "exploring forks in the path"
        elif ent < cfg.high_ent_thresh and vent > cfg.high_vent_thresh:
            temp_adj = cfg.lehv_interaction_strength_offset + cfg.lehv_interaction_strength_coef * metrics[
                "interaction_strength"]
            top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - metrics["agreement"]))))
            return self._sample(logits, temperature=min(1.5, cfg.temp * temp_adj.item()), top_p=cfg.top_p,
                                top_k=top_k_adj, min_p=cfg.min_p)

        # High Entropy, High Varentropy: "resampling in the mist"
        elif ent > cfg.med_ent_thresh and vent > cfg.high_vent_thresh:
            temp_adj = cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * metrics["attn_varentropy"]
            top_p_adj = max(0.5, cfg.top_p - cfg.hehv_attn_ent_coef * metrics["attn_entropy"].item())
            return self._sample(logits, temperature=max(2.0, cfg.temp * temp_adj.item()), top_p=top_p_adj,
                                top_k=cfg.top_k, min_p=cfg.min_p)

        # Middle ground: use adaptive sampling
        else:
            return self.adaptive_sample(logits, metrics)

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None,
                 max_length: int = None):
        max_length = max_length or self.base_model.config.max_length
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        gen_tokens = input_ids
        past_key_values = None
        position_ids = None

        for i in range(input_ids.shape[1], max_length):
            if past_key_values is None:
                inputs = {
                    'input_ids': gen_tokens,
                    'attention_mask': attention_mask,
                }
            else:
                inputs = {
                    'input_ids': gen_tokens[:, -1].unsqueeze(-1),
                    'attention_mask': attention_mask,
                    'past_key_values': past_key_values,
                }
                if position_ids is not None:
                    inputs['position_ids'] = position_ids[:, -1].unsqueeze(-1)

            outputs = self.base_model(**inputs, use_cache=True, output_attentions=True)
            logits = outputs.logits[:, -1, :]
            attention_scores = outputs.attentions
            past_key_values = outputs.past_key_values

            if position_ids is None:
                position_ids = torch.arange(gen_tokens.shape[1], device=self.device).unsqueeze(0)
            else:
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=-1)

            metrics = self.calculate_metrics(logits, attention_scores)
            next_token = self.entropy_based_sampling(gen_tokens, logits, metrics)

            gen_tokens = torch.cat([gen_tokens, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

            if next_token.item() == self.base_model.config.eos_token_id:
                break

        return gen_tokens


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with your model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", device_map="auto",
                                                      torch_dtype=torch.float16)

    # Initialize EntropyAwareModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entropy_model = EntropyAwareModel(base_model, tokenizer, device=device)

    # Prepare input
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What number is bigger 9.11 or 9.9?"},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt")

    # Generate
    output_ids = entropy_model.generate(tokenized_chat, max_length=150)

    # Decode and print the result
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Input: {messages[1]['content']}")
    print(f"Output: {output_text}")
