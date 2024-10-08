import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, GenerationConfig
from typing import List, Tuple, Dict, Optional
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
        self.last_attention_scores = None
        self._register_hooks()

    def attention_hook(self, module, input, output):
        if isinstance(output, tuple) and output[0] is not None:
            self.last_attention_scores = output[0].detach()

    def _register_hooks(self):
        if hasattr(self.model, 'model'):
            for layer in self.model.model.layers:
                layer.self_attn.register_forward_hook(self.attention_hook)
        else:
            raise ValueError("Unsupported model architecture. Expected 'model.model.layers'.")

    def calculate_varentropy_logsoftmax(self, logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(logits, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
        varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1)) ** 2, dim=axis)
        return entropy, varentropy

    def calculate_metrics(self, logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)

        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=10.0, neginf=0.0)
        varentropy = torch.nan_to_num(varentropy, nan=0.0, posinf=10.0, neginf=0.0)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
        attn_varentropy = torch.var(attn_entropy, dim=-1, unbiased=False)
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

    def adjust_scores(
            self,
            scores: torch.FloatTensor,
            temperature: float,
            top_p: float,
            top_k: int,
            min_p: float
    ) -> torch.FloatTensor:
        scores = scores / temperature
        scores = torch.nan_to_num(scores, nan=float('-inf'), posinf=100.0, neginf=-100.0)

        # Top-k filtering
        top_k = min(top_k, scores.size(-1))
        top_k_scores, _ = torch.topk(scores, top_k)
        scores[scores < top_k_scores[..., -1, None]] = float('-inf')

        # Top-p filtering
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., :top_k] = False
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        scores[indices_to_remove] = float('-inf')

        # Min-p filtering
        probs = F.softmax(scores, dim=-1)
        scores[probs < min_p] = float('-inf')

        return scores

    def adaptive_sample(self, scores: torch.FloatTensor, metrics: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        cfg = self.config
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

        logits_uncertainty = torch.nan_to_num(logits_uncertainty, nan=0.0, posinf=10.0, neginf=0.0)
        attn_uncertainty = torch.nan_to_num(attn_uncertainty, nan=0.0, posinf=10.0, neginf=0.0)

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

        return self.adjust_scores(scores, temperature.item(), top_p.item(), top_k, min_p.item())

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.last_attention_scores is None:
            print("Warning: No attention scores available. Using default sampling.")
            return scores

        metrics = self.calculate_metrics(scores, self.last_attention_scores)
        ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
        cfg = self.config

        print(f"Entropy: {ent.item():.4f}, Varentropy: {vent.item():.4f}")
        print(f"Attention Entropy: {metrics['attn_entropy'].item():.4f}, Attention Varentropy: {metrics['attn_varentropy'].item():.4f}")

        # Low Entropy, Low Varentropy: "flowing with unspoken intent"
        if ent < cfg.low_ent_thresh and vent < cfg.low_vent_thresh:
            print("Low entropy and varentropy: using top-k sampling")
            return self.adjust_scores(scores, cfg.temp, 1.0, 5, 0.0)

        # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
        elif ent > cfg.high_ent_thresh and vent < cfg.low_vent_thresh:
            print("High entropy, low varentropy")
            if not torch.isin(input_ids[:, -1], torch.tensor([self.clarifying_question_token_id])).any():
                print("Inserting clarifying question token")
                scores.fill_(float('-inf'))
                scores[:, self.clarifying_question_token_id] = 0
            else:
                temp_adj = cfg.helv_attn_ent_offset + cfg.helv_attn_ent_coef * metrics["attn_entropy"]
                print(f"Adjusted temperature: {temp_adj:.4f}")
                return self.adjust_scores(scores, min(1.5, cfg.temp * temp_adj.item()), cfg.top_p, cfg.top_k, cfg.min_p)

        # Low Entropy, High Varentropy: "exploring forks in the path"
        elif ent < cfg.high_ent_thresh and vent > cfg.high_vent_thresh:
            print("Low entropy, high varentropy")
            temp_adj = cfg.lehv_interaction_strength_offset + cfg.lehv_interaction_strength_coef * metrics["interaction_strength"]
            top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - metrics["agreement"]))))
            print(f"Adjusted temperature: {temp_adj:.4f}, Adjusted top_k: {top_k_adj}")
            return self.adjust_scores(scores, min(1.5, cfg.temp * temp_adj.item()), cfg.top_p, top_k_adj, cfg.min_p)

        # High Entropy, High Varentropy: "resampling in the mist"
        elif ent > cfg.med_ent_thresh and vent > cfg.high_vent_thresh:
            print("High entropy and varentropy")
            temp_adj = cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * metrics["attn_varentropy"]
            top_p_adj = max(0.5, cfg.top_p - cfg.hehv_attn_ent_coef * metrics["attn_entropy"].item())
            print(f"Adjusted temperature: {temp_adj:.4f}, Adjusted top_p: {top_p_adj:.4f}")
            return self.adjust_scores(scores, max(1.5, cfg.temp * temp_adj.item()), top_p_adj, cfg.top_k, cfg.min_p)

        # Middle ground: use adaptive sampling
        else:
            print("Using adaptive sampling")
            return self.adaptive_sample(scores, metrics)


# Example usage
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Assuming "..." is our pause token and "?" is our clarifying question token
pause_token_id = tokenizer.convert_tokens_to_ids("...")
clarifying_question_token_id = tokenizer.convert_tokens_to_ids("?")

entropy_sampler = HookBasedEntropySampler(model, tokenizer, pause_token_id, clarifying_question_token_id)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What number is bigger 9.11 or 9.9?"},
]
# Tokenize the chat and create attention mask
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
attention_mask = tokenized_chat.ne(tokenizer.eos_token_id).long()

# Set up the generation configuration
generation_config = GenerationConfig(
    max_new_tokens=500,
    do_sample=True,
    temperature=0.45,
    top_p=0.95,
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
