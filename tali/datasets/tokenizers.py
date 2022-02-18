import os
import warnings

import torch
from torch import nn
from transformers import CLIPTokenizerFast

from tali.base import utils

log = utils.get_logger(__name__)


class HuggingFaceBPETokenizer(nn.Module):
    def __init__(self, context_length):
        super(HuggingFaceBPETokenizer, self).__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = CLIPTokenizerFast.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        self.context_length = context_length

    def forward(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokenized_words = self.tokenizer(x)

        tokenized_tensor = torch.Tensor(tokenized_words)
        preshape = tokenized_tensor.shape
        if len(tokenized_tensor.shape) == 2:
            tokenized_tensor = tokenized_tensor.view(-1)
        postshape = tokenized_tensor.shape

        if len(tokenized_tensor) > self.context_length:
            return tokenized_tensor[: self.context_length]

        diff_length = self.context_length - len(tokenized_tensor)
        try:
            padding_tensor = torch.zeros(diff_length)
            return torch.cat([tokenized_tensor, padding_tensor], dim=0)
        except Exception:
            log.error(f"Error bro {preshape} {postshape}.")

    def batch_decode(self, x):
        return self.tokenizer.batch_decode(
            x, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def decode(self, x):
        return self.tokenizer.decode(x)
