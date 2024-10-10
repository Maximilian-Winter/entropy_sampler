import torch
import unittest
from entropy_sampler import HookBasedEntropySampler, SamplerConfig

class MockModel:
    def __init__(self):
        self.model = type('MockInnerModel', (), {'layers': [type('MockLayer', (), {'self_attn': type('MockAttention', (), {'register_forward_hook': lambda *args: None})})]()})

class MockTokenizer:
    def __init__(self):
        pass

class TestHookBasedEntropySampler(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.tokenizer = MockTokenizer()
        self.config = SamplerConfig()
        self.sampler = HookBasedEntropySampler(
            self.model,
            self.tokenizer,
            pause_token_id=1000,
            clarifying_question_token_id=2000,
            config=self.config
        )

    def test_low_entropy_low_varentropy(self):
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 10)
        self.sampler.last_attention_scores = torch.randn(1, 1, 10, 10)
        
        # Mock low entropy and low varentropy
        self.sampler.calculate_metrics = lambda *args: {
            "logits_entropy": torch.tensor(0.05),
            "logits_varentropy": torch.tensor(0.05),
            "attn_entropy": torch.tensor(0.1),
            "attn_varentropy": torch.tensor(0.1),
            "agreement": torch.tensor(0.9),
            "interaction_strength": torch.tensor(0.1)
        }

        result = self.sampler(input_ids, scores)
        self.assertIsInstance(result, torch.FloatTensor)
        self.assertEqual(result.shape, scores.shape)

    def test_high_entropy_low_varentropy(self):
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 10)
        self.sampler.last_attention_scores = torch.randn(1, 1, 10, 10)
        
        # Mock high entropy and low varentropy
        self.sampler.calculate_metrics = lambda *args: {
            "logits_entropy": torch.tensor(6.0),
            "logits_varentropy": torch.tensor(0.05),
            "attn_entropy": torch.tensor(5.0),
            "attn_varentropy": torch.tensor(0.1),
            "agreement": torch.tensor(0.3),
            "interaction_strength": torch.tensor(0.5)
        }

        result = self.sampler(input_ids, scores)
        self.assertIsInstance(result, torch.FloatTensor)
        self.assertEqual(result.shape, scores.shape)

    # Add more test cases for other scenarios...

if __name__ == '__main__':
    unittest.main()
