from typing import List, Union

import torch


def tokenize(
    sequence_of_text: Union[str, List[str]],
    tokenizer: object,
    context_length: int = 77,
) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    sequence_of_text : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input
    strings, context_length]
    :param tokenizer:
    """
    if isinstance(sequence_of_text, str):
        sequence_of_text = [sequence_of_text]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [
        [sot_token] + tokenizer.encode(text) + [eot_token] for text in sequence_of_text
    ]
    result = torch.zeros(size=(len(all_tokens), context_length)).long()

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]

        result[i, : len(tokens)] = torch.tensor(tokens)

    return result
