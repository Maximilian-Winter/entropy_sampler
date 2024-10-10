import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, Tuple, Optional

LN_2 = 0.69314718056  # ln(2)

def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
    return entropy, varentropy

def multinomial_sample_one(probs_sort: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    q = torch.rand(probs_sort.shape, generator=generator, device=probs_sort.device)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(torch.int32)

def _sample(logits: torch.Tensor, temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = F.softmax(logit / temperature, dim=-1)

    # Apply min_p sampling
    if min_p > 0.0:
        p_max = torch.max(probs, dim=-1, keepdim=True).values
        indices_to_remove = probs < (min_p * p_max)
        logit = torch.where(indices_to_remove, torch.full_like(logit, float('-inf')), logit)
        probs = F.softmax(logit, dim=-1)

    # Apply top-k sampling
    top_k = min(top_k, probs.shape[-1])
    if top_k > 0:
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
    else:
        top_k_probs, top_k_indices = probs, torch.arange(probs.shape[-1], device=probs.device).unsqueeze(0).expand(bsz, -1)
    probs_sort = top_k_probs
    probs_idx = top_k_indices
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Apply top-p sampling
    mask = probs_sum > top_p
    probs_sort = probs_sort.masked_fill(mask, 0)
    probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)
    next_token = multinomial_sample_one(probs_sort, generator)
    # Convert next_token to int64 before using it in gather
    next_token_g = torch.gather(probs_idx, -1, next_token.reshape(bsz, 1).to(torch.int64))
    return next_token_g.to(torch.int32)

def calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)
    # attention_scores shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
    # Average over layers
    attention_scores = torch.stack(attention_scores) if isinstance(attention_scores, list) else attention_scores
    attention_scores = torch.mean(attention_scores, dim=0)  # Shape: (batch_size, num_heads, seq_len, seq_len)
    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
    attn_varentropy = torch.var(attn_entropy, dim=-1)

    # Add a small epsilon to avoid NaN when all values are the same
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

def adaptive_sample(logits: torch.Tensor, metrics: Dict[str, torch.Tensor],
                    gen_tokens: torch.Tensor, n_samples: int,
                    base_temp: float = 0.666, base_top_p: float = 0.90, base_top_k: int = 40, base_min_p: float = 0.03,
                    generator: Optional[torch.Generator] = None) -> torch.Tensor:
    logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
    attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

    temperature = base_temp * (1 + 0.3 * logits_uncertainty + 0.2 * attn_uncertainty - 0.2 * metrics["agreement"])
    top_p = torch.clamp(base_top_p * (1 + 0.1 * metrics["attn_varentropy"]), 0.1, 1.0)
    top_k = int(torch.clamp(
        torch.round(torch.tensor(base_top_k) * (1 + 0.3 * metrics["interaction_strength"].item() - 0.2 * metrics["agreement"].item())),
        min=1,
        max=100
    ).item())
    min_p = torch.clamp(base_min_p * (1 - 0.5 * logits_uncertainty), 0.01, 0.5)

    samples = []
    for _ in range(n_samples):
        sample = _sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, generator=generator)
        samples.append(sample)

    def score_sample(sample):
        # Flatten the sample tensor and convert to long (int64)
        sample_flat = sample.flatten().to(torch.long)

        # Create one-hot encoding
        one_hot = F.one_hot(sample_flat, logits.shape[-1])

        # Reshape log_softmax output to match one_hot
        log_probs = F.log_softmax(logits[:, -1], dim=-1)  # Only consider the last token's logits
        log_probs = log_probs.view(-1, logits.shape[-1])

        # Calculate log probability
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

def sample(gen_tokens: torch.Tensor, logits: torch.Tensor, attention_scores: torch.Tensor,
           temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0,
           generator: Optional[torch.Generator] = None) -> torch.Tensor:
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if ent < 0.1 and vent < 0.1:
        return torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif ent > 3.0 and vent < 0.1:
        # Insert a clarifying question token if not already present
        if not torch.isin(gen_tokens[:, -1], torch.tensor([2564], device=gen_tokens.device)).any():
            return torch.tensor([[2564]], dtype=torch.int32, device=gen_tokens.device)  # Adjust as needed
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = 1.3 + 0.2 * attn_ent  # Increase temperature based on attention entropy
            return _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k, min_p=min_p, generator=generator)

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif ent < 5.0 and vent > 5.0:
        temp_adj = 1.2 + 0.3 * interaction_strength  # Increase temperature based on interaction strength
        top_k_adj = max(5, int(top_k * (1 + 0.5 * (1 - agreement))))  # Increase top_k when agreement is low
        return _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k_adj, min_p=min_p, generator=generator)

    # High Entropy, High Varentropy: "resampling in the mist"
    elif ent > 5.0 and vent > 5.0:
        # Use high temperature and adjusted top_p based on attention metrics
        temp_adj = 2.0 + 0.5 * attn_vent  # Increase temperature based on attention varentropy
        top_p_adj = max(0.5, top_p - 0.2 * attn_ent)  # Decrease top_p when attention entropy is high
        return _sample(logits, temperature=max(2.0, temperature * temp_adj), top_p=top_p_adj, top_k=top_k, min_p=min_p, generator=generator)

    # Middle ground: use adaptive sampling
    else:
        return adaptive_sample(
            logits,
            metrics,
            gen_tokens,
            n_samples=5,
            base_temp=temperature,
            base_top_p=top_p,
            base_top_k=top_k,
            generator=generator
        )

class EntropyBasedSamplingModel:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model.to(self.device)

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                 temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0,
                 max_length: int = 50, generator: Optional[torch.Generator] = None):
        generated = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        else:
            attention_mask = torch.ones_like(generated, device=self.device)

        past_key_values = None
        for _ in range(max_length):
            if past_key_values is None:
                # For the first step, pass the entire input_ids
                inputs = generated
                current_attention_mask = attention_mask
            else:
                # For subsequent steps, only pass the last token
                inputs = generated[:, -1].unsqueeze(-1)
                current_attention_mask = attention_mask[:, -1].unsqueeze(-1)

            outputs = self.model(
                input_ids=inputs,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                output_attentions=True,
                use_cache=True,
                return_dict=True
            )
            logits = outputs.logits  # Shape: (batch_size, seq_len=1, vocab_size)
            attention_scores = list(outputs.attentions)  # List of tensors

            # Now call the sample function
            next_token = sample(generated, logits, attention_scores,
                                temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, generator=generator)
            generated = torch.cat((generated, next_token), dim=1)
            # Update past_key_values for faster generation
            past_key_values = outputs.past_key_values
            # Update attention_mask
            attention_mask = torch.cat(
                (attention_mask, torch.ones((attention_mask.shape[0], 1), device=self.device, dtype=attention_mask.dtype)),
                dim=1
            )
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        return generated



from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",attn_implementation="eager", return_dict_in_generate=False, output_attentions=True).to("cuda")

# Initialize the sampler
pause_token_id = 128237
clarifying_question_token_id = 128238
entropy_aware_model = EntropyBasedSamplingModel(model, tokenizer,
                                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Prepare input
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What number is bigger 9.11 or 9.9?"},
]
tokenized_chat = tokenizer.apply_chat_template(messages,  tokenize=True, add_generation_prompt=True, return_tensors="pt")
attention_mask = tokenized_chat.ne(tokenizer.eos_token_id).long()

# Generate output
output_ids = entropy_aware_model.generate(
    tokenized_chat,
    attention_mask
)

# Decode and print the result
print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
