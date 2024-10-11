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

        logger.info(f"Initialized EntropyAwareModel using device: {self.device}")

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def calculate_varentropy_logsoftmax(self, logits: torch.Tensor, axis: int = -1) -> Tuple[
        torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(logits, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / math.log(2)
        varentropy = torch.sum(probs * (log_probs / math.log(2) + entropy.unsqueeze(-1)) ** 2, dim=axis)
        return entropy, varentropy

    def calculate_metrics(self, logits: torch.Tensor, attention_scores: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)

        # Process attention scores
        attn_scores = torch.stack(attention_scores, dim=0)
        current_token_attn = attn_scores[:, :, :, -1, :]  # [num_layers, batch_size, num_heads, seq_len]

        attention_probs = F.softmax(current_token_attn, dim=-1)
        attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
        attn_varentropy = torch.var(attn_entropy, dim=-1, unbiased=False)

        # Calculate layer-wise metrics
        layer_metrics = []
        for layer in range(attn_scores.size(0)):
            layer_attn = attention_probs[layer]
            layer_mean_attention = torch.mean(layer_attn, dim=1)
            layer_agreement = torch.mean(torch.abs(layer_attn - layer_mean_attention.unsqueeze(1)), dim=(1, 2))
            layer_interaction_strength = torch.mean(torch.abs(current_token_attn[layer]))
            layer_metrics.append({
                "agreement": layer_agreement,
                "interaction_strength": layer_interaction_strength
            })

        # Aggregate metrics across layers
        agreement = torch.mean(torch.stack([lm["agreement"] for lm in layer_metrics]))
        interaction_strength = torch.mean(torch.stack([lm["interaction_strength"] for lm in layer_metrics]))

        return {
            "logits_entropy": torch.mean(entropy),
            "logits_varentropy": torch.mean(varentropy),
            "attn_entropy": torch.mean(attn_entropy),
            "attn_varentropy": torch.mean(attn_varentropy),
            "agreement": agreement,
            "interaction_strength": interaction_strength,
            "layer_metrics": layer_metrics
        }

    def adjust_scores(self, scores: torch.FloatTensor, temperature: float, top_p: float, top_k: int,
                      min_p: float) -> torch.FloatTensor:
        scores = scores / temperature
        scores = torch.nan_to_num(scores, nan=float('-inf'), posinf=100.0, neginf=-100.0)

        top_k = min(top_k, scores.size(-1))
        top_k_scores, _ = torch.topk(scores, top_k)
        scores[scores < top_k_scores[..., -1, None]] = float('-inf')

        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., :top_k] = False
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        scores[indices_to_remove] = float('-inf')

        probs = F.softmax(scores, dim=-1)
        scores[probs < min_p] = float('-inf')

        return scores

    def adaptive_sample(self, scores: torch.FloatTensor, metrics: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        cfg = self.config
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

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

    def entropy_based_sampling(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                               metrics: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
        cfg = self.config

        low_ent_thresh = cfg.low_ent_thresh
        high_ent_thresh = cfg.high_ent_thresh
        low_vent_thresh = cfg.low_vent_thresh
        high_vent_thresh = cfg.high_vent_thresh

        logger.info(f"Entropy: {ent.item():.4f}, Varentropy: {vent.item():.4f}")
        logger.info(
            f"Attention Entropy: {metrics['attn_entropy'].item():.4f}, Attention Varentropy: {metrics['attn_varentropy'].item():.4f}")

        # Sampling logic
        if ent < low_ent_thresh and vent < low_vent_thresh:
            logger.info("Low entropy and varentropy: using greedy sampling")
            next_token = torch.argmax(scores, dim=-1, keepdim=True)
            return next_token
        elif ent > high_ent_thresh and vent < low_vent_thresh:
            logger.info("High entropy, low varentropy")
            clarifying_question_token_id = 128238
            if not torch.isin(input_ids[:, -1], torch.tensor([clarifying_question_token_id], device=self.device)).any():
                logger.info("Inserting clarifying question token")
                scores.fill_(float('-inf'))
                scores[:, clarifying_question_token_id] = 0
                return scores
            else:
                temp_adj = cfg.helv_attn_ent_offset + cfg.helv_attn_ent_coef * metrics["attn_entropy"]
                logger.info(f"Adjusted temperature: {temp_adj:.4f}")
                return self.adjust_scores(scores, min(1.5, cfg.temp * temp_adj.item()), cfg.top_p, cfg.top_k, cfg.min_p)
        elif ent < high_ent_thresh and vent > high_vent_thresh:
            logger.info("Low entropy, high varentropy")
            temp_adj = cfg.lehv_interaction_strength_offset + cfg.lehv_interaction_strength_coef * metrics[
                "interaction_strength"]
            top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - metrics["agreement"]))))
            logger.info(f"Adjusted temperature: {temp_adj:.4f}, Adjusted top_k: {top_k_adj}")
            return self.adjust_scores(scores, min(1.5, cfg.temp * temp_adj.item()), cfg.top_p, top_k_adj, cfg.min_p)
        elif ent > cfg.med_ent_thresh and vent > high_vent_thresh:
            logger.info("High entropy and varentropy")
            temp_adj = cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * metrics["attn_varentropy"]
            top_p_adj = max(0.5, cfg.top_p - cfg.hehv_attn_ent_coef * metrics["attn_entropy"].item())
            logger.info(f"Adjusted temperature: {temp_adj:.4f}, Adjusted top_p: {top_p_adj:.4f}")
            return self.adjust_scores(scores, max(1.5, cfg.temp * temp_adj.item()), top_p_adj, cfg.top_k, cfg.min_p)
        else:
            logger.info("Using adaptive sampling")
            return self.adaptive_sample(scores, metrics)

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None, max_length: int = None):
        max_length = max_length or self.base_model.config.max_length
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)



        for i in range(input_ids.shape[1], max_length):
            # Prepare inputs
            if i == input_ids.shape[1]:
                inputs = self.base_model.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask)
            else:
                inputs = self.base_model.prepare_inputs_for_generation(input_ids[:, -1:], past_key_values=past_key_values)

            # Forward pass
            outputs = self.base_model(**inputs, output_attentions=True)
            logits = outputs.logits[:, -1, :]
            attention_scores = outputs.attentions
            past_key_values = outputs.past_key_values

            # Calculate metrics and sample next token
            metrics = self.calculate_metrics(logits, attention_scores)
            adjusted_logits = self.entropy_based_sampling(input_ids, logits, metrics)
            if adjusted_logits.shape[1] == 1:
                next_token = adjusted_logits
            else:
                next_token = torch.multinomial(F.softmax(adjusted_logits, dim=-1), num_samples=1)

            # Append new token to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

            # Check for EOS token
            if next_token.item() == self.base_model.config.eos_token_id:
                break

        return input_ids


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with your model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", device_map="auto", torch_dtype=torch.float16)

    # Initialize EntropyAwareModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entropy_model = EntropyAwareModel(base_model, tokenizer, device=device)

    # Prepare input
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What number is bigger 9.11 or 9.9? Let's think through this step by step."},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt")

    # Generate
    output_ids = entropy_model.generate(tokenized_chat, max_length=500)

    # Decode and print the result
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(f"Input: {messages[1]['content']}")
    print(f"Output: {output_text}")
    output_ids = base_model.generate(tokenized_chat.to("cuda"), max_length=500)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(f"Input: {messages[1]['content']}")
    print(f"Output: {output_text}")