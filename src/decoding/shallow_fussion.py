import torch
from transformers import AutoModelForCausalLM, LogitsProcessor


class LMRescorerLogitsProcessor(LogitsProcessor):
    """Logits Processor to rescore the next token scores with a language model."""

    def __init__(self, lm_weight: float, lm_model: str, device: torch.device):
        super().__init__()
        self.lm_model = AutoModelForCausalLM.from_pretrained(lm_model).to(device)
        self.lm_weight = lm_weight

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = self.lm_model(input_ids).logits
        lm_scores = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)
        next_token_scores = scores + self.lm_weight * lm_scores
        return next_token_scores
