"""Fast truncation of strings to the maximum token length, using tiktoken."""

from typing import Tuple

import tiktoken
from tiktoken import Encoding
from typeguard import typechecked

from tiktoken_truncate.globals import model_max_tokens
from tiktoken_truncate.random_string import random_string

# Cache for average tokens per character
avg_tokens_per_char_cache = {}


def estimate_avg_tokens_per_char(encoding: Encoding) -> float:
    """Estimate the average number of tokens per character, using a random string."""
    sample_text = random_string(1024)
    tokens = encoding.encode(sample_text)
    avg = len(tokens) / len(sample_text)
    print(f"Estimated average tokens per character: {avg} for {encoding}")
    return avg


def get_avg_tokens_per_char(encoding: Encoding) -> float:
    """Get the average tokens per character, cached."""
    global avg_tokens_per_char_cache
    if encoding not in avg_tokens_per_char_cache:
        avg_tokens_per_char_cache[encoding] = estimate_avg_tokens_per_char(encoding)
    return avg_tokens_per_char_cache[encoding]


def find_bounds(
    text: str, encoding: Encoding, max_tokens: int, avg_tokens_per_char: float
) -> Tuple[int, int]:
    """Find the initial bounds for binary search."""
    estimated_length = int(max_tokens / avg_tokens_per_char)
    low, high = None, None

    high_estimated_length = estimated_length
    while high is None:
        high_estimated_length = max(int(high_estimated_length * 1.1), high_estimated_length + 10)
        if high_estimated_length >= len(text):
            high = len(text)
        high_ntokens = len(encoding.encode(text[:high_estimated_length]))
        print(f"high_estimated_length: {high_estimated_length}, high_ntokens: {high_ntokens}")
        if high_ntokens > max_tokens:
            high = high_estimated_length
            low = high_estimated_length
            return low, high

    low_estimated_length = estimated_length
    while low is None:
        low_estimated_length = min(int(low_estimated_length / 1.1), low_estimated_length - 10)
        low_ntokens = len(encoding.encode(text[:low_estimated_length]))
        if low_ntokens < max_tokens:
            low = low_estimated_length

    return low, high


@typechecked
def truncate_document_to_max_tokens(text: str, model: str) -> str:
    """Fast truncation of strings to the maximum token length, using tiktoken."""
    print(f"truncate_document_to_max_tokens: {len(text)} characters")
    max_tokens = model_max_tokens[model]
    encoding = tiktoken.encoding_for_model(model)

    avg_tokens_per_char = get_avg_tokens_per_char(encoding)

    # Estimate initial bounds without encoding the entire text
    _, high = find_bounds(text, encoding, max_tokens, avg_tokens_per_char)

    truelen = high
    # for truelen in range(high, low, -1): # Maybe down to 1?
    for truelen in range(high, 1, -1):  # Maybe down to 1?
        if len(encoding.encode(text[:truelen])) <= max_tokens:
            break

    print(
        f"FAST Estimated true length: {len(encoding.encode(text[:truelen]))} "
        f"for {truelen} characters"
    )

    return text[:truelen]
