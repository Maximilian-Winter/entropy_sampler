import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Optional, Tuple, Dict, NamedTuple


class KVCache(nn.Module):
    def __init__(self, layers: int, max_seq_len: int):
        super(KVCache, self).__init__()
        self.layers = layers
        self.max_seq_len = max_seq_len
        self.k_cache = [None] * layers
        self.v_cache = [None] * layers

    def update(self, xk: torch.Tensor, xv: torch.Tensor, layer_idx: int, cur_pos: int):
        # xk and xv shapes: (batch_size, num_heads, seq_len, head_dim)
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = xk
            self.v_cache[layer_idx] = xv
        else:
            if cur_pos + xk.size(2) > self.max_seq_len:
                # If we exceed max_seq_len, we need to slice off the beginning
                excess = cur_pos + xk.size(2) - self.max_seq_len
                self.k_cache[layer_idx] = self.k_cache[layer_idx][:, :, excess:]
                self.v_cache[layer_idx] = self.v_cache[layer_idx][:, :, excess:]
                cur_pos = self.max_seq_len - xk.size(2)

            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], xk], dim=2)[:, :, :self.max_seq_len]
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], xv], dim=2)[:, :, :self.max_seq_len]

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def clear(self):
        for i in range(self.layers):
            self.k_cache[i] = None
            self.v_cache[i] = None


class AttnStats(NamedTuple):
    entropy: torch.Tensor
    varentropy: torch.Tensor
    n_layers: int
    n_heads: int

    @classmethod
    def new(cls, bsz: int, n_layers: int, n_heads: int) -> 'AttnStats':
        return cls(
            entropy=torch.zeros((bsz, n_layers, n_heads), dtype=torch.float32,
                                device='cuda' if torch.cuda.is_available() else 'cpu'),
            varentropy=torch.zeros((bsz, n_layers, n_heads), dtype=torch.float32,
                                   device='cuda' if torch.cuda.is_available() else 'cpu'),
            n_layers=n_layers,
            n_heads=n_heads
        )

    @property
    def avg_entropy(self):
        return self.entropy.sum(dim=-1, keepdim=False)

    @property
    def std_error(self):
        return torch.sqrt(torch.mean(self.varentropy)) / (self.n_heads * self.n_layers)

    def update(self, scores: torch.Tensor, layer_idx: int):
        probs = F.softmax(scores, dim=-1)
        new_entropy = -torch.sum(torch.where(probs > 0, probs * torch.log(probs), torch.tensor(0.0)), dim=-1)
        new_varentropy = torch.sum(probs * (torch.log(probs) + new_entropy.unsqueeze(-1)) ** 2, dim=-1)

        self.entropy[:, layer_idx, :] = new_entropy
        self.varentropy[:, layer_idx, :] = new_varentropy

        return self


class EntropyBasedSampler:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Dict[str, any]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.config = config

        # Initialize KVCache
        self.kvcache = KVCache(
            layers=config['n_layers'],
            max_seq_len=config['max_seq_len']
        ).to(self.device)

    @staticmethod
    def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        LN_2 = 0.69314718056
        log_probs = F.log_softmax(logits, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2
        varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1)) ** 2, dim=axis)
        return entropy, varentropy

    @staticmethod
    def calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        entropy, varentropy = EntropyBasedSampler.calculate_varentropy_logsoftmax(logits)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
        attn_varentropy = torch.var(attn_entropy, dim=-1)

        attn_varentropy = torch.where(torch.isnan(attn_varentropy), torch.zeros_like(attn_varentropy), attn_varentropy)
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
    def _sample(logits: torch.Tensor, temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0,
                generator: torch.Generator = None) -> torch.Tensor:
        bsz = logits.shape[0]
        logit = logits[:, -1]
        probs = F.softmax(logit / temperature, dim=-1)

        if min_p > 0.0:
            p_max = torch.max(probs, dim=-1, keepdim=True).values
            indices_to_remove = probs < (min_p * p_max)
            logit = torch.where(indices_to_remove, torch.full_like(logit, float('-inf')), logit)

        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        probs_sort = torch.flip(top_k_probs, dims=[-1])
        probs_idx = torch.flip(top_k_indices, dims=[-1])
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = torch.where(probs_sum - probs_sort > top_p, torch.tensor(1.0, device=probs.device),
                           torch.tensor(0.0, device=probs.device))
        probs_sort = probs_sort * (1 - mask)
        probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)
        next_token = torch.multinomial(probs_sort, num_samples=1, generator=generator)
        next_token_g = torch.gather(probs_idx, -1, next_token.reshape(bsz, 1).to(torch.int64))
        return next_token_g.to(torch.int32)

    def adaptive_sample(self, logits: torch.Tensor, metrics: Dict[str, torch.Tensor],
                        gen_tokens: torch.Tensor, n_samples: int,
                        base_temp: float = 0.666, base_top_p: float = 0.90, base_top_k: int = 40,
                        base_min_p: float = 0.03,
                        generator: torch.Generator = None) -> torch.Tensor:
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

        temperature = base_temp * (1 + 0.3 * logits_uncertainty + 0.2 * attn_uncertainty - 0.2 * metrics["agreement"])
        top_p = torch.clamp(base_top_p * (1 + 0.1 * metrics["attn_varentropy"]), 0.1, 1.0)
        top_k = int(torch.clamp(
            torch.round(torch.tensor(base_top_k) * (
                    1 + 0.3 * metrics["interaction_strength"].item() - 0.2 * metrics["agreement"].item())),
            min=1,
            max=100
        ).item())
        min_p = torch.clamp(base_min_p * (1 - 0.5 * logits_uncertainty), 0.01, 0.5)

        samples = []
        for _ in range(n_samples):
            sample = self._sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
                                  generator=generator)
            samples.append(sample)

        def score_sample(sample):
            sample_flat = sample.flatten().to(torch.long)
            one_hot = F.one_hot(sample_flat, logits.shape[-1])
            log_probs = F.log_softmax(logits, dim=-1).view(-1, logits.shape[-1])
            log_prob = torch.sum(log_probs * one_hot)

            confidence_score = (
                    (1 - metrics["logits_entropy"]) * 0.1 +
                    (1 - metrics["attn_entropy"]) * 0.2 +
                    (1 - metrics["logits_varentropy"]) * 0.3 +
                    (1 - metrics["attn_varentropy"]) * 0.4 +
                    metrics["agreement"] * 0.5 +
                    metrics["interaction_strength"] * 0.6
            )
            return log_prob + confidence_score

        sample_scores = torch.stack([score_sample(sample) for sample in samples])
        best_sample_idx = torch.argmax(sample_scores)
        return samples[best_sample_idx]

    def sample(self, gen_tokens: torch.Tensor, logits: torch.Tensor, attention_scores: torch.Tensor,
               temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.1,
               generator: torch.Generator = None) -> torch.Tensor:
        metrics = self.calculate_metrics(logits, attention_scores)
        ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
        attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
        agreement = metrics["agreement"]
        interaction_strength = metrics["interaction_strength"]

        if ent < 0.1 and vent < 0.1:
            return torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
        elif ent > 3.0 and vent < 0.1:
            if not torch.isin(gen_tokens[:, -1], torch.tensor([2564], device=self.device)).any():
                return torch.tensor([[2564]], dtype=torch.int32, device=self.device)
            else:
                temp_adj = 1.3 + 0.2 * attn_ent
                return self._sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k,
                                    min_p=min_p, generator=generator)
        elif ent < 5.0 and vent > 5.0:
            temp_adj = 1.2 + 0.3 * interaction_strength
            top_k_adj = max(5, int(top_k * (1 + 0.5 * (1 - agreement))))
            return self._sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k_adj,
                                min_p=min_p, generator=generator)
        elif ent > 5.0 and vent > 5.0:
            temp_adj = 2.0 + 0.5 * attn_vent
            top_p_adj = max(0.5, top_p - 0.2 * attn_ent)
            return self._sample(logits, temperature=max(2.0, temperature * temp_adj), top_p=top_p_adj, top_k=top_k,
                                min_p=min_p, generator=generator)
        else:
            return self.adaptive_sample(
                logits,
                metrics,
                gen_tokens,
                n_samples=5,
                base_temp=temperature,
                base_top_p=top_p,
                base_top_k=top_k,
                generator=generator
            )

    def generate(self, input_ids: torch.Tensor, max_length: int, temperature: float = 0.666, top_p: float = 0.90,
                 top_k: int = 27, min_p: float = 0.0) -> List[int]:
        generator = torch.Generator(device=self.device).manual_seed(self.config['seed'])
        gen_tokens = input_ids.clone()
        attn_stats = AttnStats.new(1, self.config['n_layers'], self.config['n_heads'])
        self.kvcache.clear()  # Clear KVCache before generation

        for cur_pos in range(max_length - len(input_ids[0])):
            inputs = {'input_ids': gen_tokens[:, -1:], 'return_dict': True, 'output_attentions': True, 'use_cache': True}
            if cur_pos > 0:
                inputs['past_key_values'] = past_key_values

            try:
                outputs = self.model(**inputs)
            except RuntimeError as e:
                print(f"Error during generation at position {cur_pos}: {e}")
                break

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # Update attention statistics
            for layer_idx, layer_attention in enumerate(outputs.attentions):
                attn_stats = attn_stats.update(layer_attention[:, :, -1, :], layer_idx)

            # Update KVCache
            for layer_idx, (k, v) in enumerate(past_key_values):
                self.kvcache.update(k, v, layer_idx, cur_pos)

            # Use the last layer's attention scores for sampling
            attention_scores = outputs.attentions

            next_token = self.sample(gen_tokens, logits, attention_scores, temperature, top_p, top_k, min_p, generator)

            if self.config['verbose']:
                decoded_token = self.tokenizer.decode(next_token.squeeze())
                print(decoded_token, end="", flush=True)

            gen_tokens = torch.cat((gen_tokens, next_token), dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return gen_tokens[0].tolist()


from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to("cuda")
config_dict = {
    'n_layers': model.config.num_hidden_layers,
    'n_heads': model.config.num_attention_heads,
    'head_dim': model.config.head_dim,
    'max_seq_len': 2048,
    'seed': 1337,
    'verbose': True
}
entropy_sampler_model = EntropyBasedSampler(model, tokenizer, config_dict)

# Prepare input
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What number is bigger 9.11 or 9.9?"},
]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

output_ids = entropy_sampler_model.generate(
    tokenized_chat.to("cuda"), max_length=250
)

# Decode and print the result
print(tokenizer.decode(output_ids[0], skip_special_tokens=False))