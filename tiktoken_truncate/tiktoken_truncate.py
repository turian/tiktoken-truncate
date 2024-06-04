"""Fast speed truncation of strings to the maximum token length, using tiktoken."""

from typing import Dict

import tiktoken
from tiktoken import Encoding
from typeguard import typechecked

from tiktoken_truncate.globals import model_max_tokens
from tiktoken_truncate.random_string import random_string

# Cache for average tokens per character
avg_tokens_per_char_cache: Dict[Encoding, float] = {}


@typechecked
def estimate_avg_tokens_per_char(encoding: Encoding) -> float:
    """Estimate the average number of tokens per character, using a random string."""
    sample_text = random_string(1024)
    tokens = encoding.encode(sample_text)
    avg = len(tokens) / len(sample_text)
    return avg


@typechecked
def get_avg_tokens_per_char(encoding: Encoding) -> float:
    """Get the average tokens per character, cached."""
    global avg_tokens_per_char_cache
    if encoding not in avg_tokens_per_char_cache:
        avg_tokens_per_char_cache[encoding] = estimate_avg_tokens_per_char(encoding)
    return avg_tokens_per_char_cache[encoding]


@typechecked
def cached_encode_length(
    encoding_cache: Dict[int, int], encoding: Encoding, text: str, length: int
) -> int:
    """Return the number of tokens for the given text length, using cache."""
    if length > len(text):
        raise AssertionError(f"length ({length}) is greater than text ({len(text)})")
    if length not in encoding_cache:
        encoding_cache[length] = len(encoding.encode(text[:length]))
    return encoding_cache[length]


@typechecked
def expand_high(
    encoding_cache: Dict[int, int], encoding: Encoding, text: str, high: int, max_tokens: int
) -> int:
    """Expand the high bound until the token count exceeds max_tokens."""
    while cached_encode_length(encoding_cache, encoding, text, high) <= max_tokens:
        high = max(int(high * 1.1), high + 1)
        if high >= len(text):
            high = len(text)
            break
    return high


@typechecked
def expand_low(
    encoding_cache: Dict[int, int], encoding: Encoding, text: str, low: int, max_tokens: int
) -> int:
    """Expand the low bound until the token count is at most max_tokens."""
    while cached_encode_length(encoding_cache, encoding, text, low) > max_tokens:
        low = min(int(low / 1.1), low - 1)
    if low <= 0:
        raise AssertionError(f"low ({low}) is less than or equal to 0")
    if low > len(text):
        raise AssertionError(f"low ({low}) is greater than len(text) ({len(text)})")
    return low


@typechecked
def binary_search_max_length(
    encoding_cache: Dict[int, int],
    encoding: Encoding,
    text: str,
    low: int,
    high: int,
    max_tokens: int,
) -> int:
    """Use binary search to find the maximal length where the token count is <= max_tokens."""
    if cached_encode_length(encoding_cache, encoding, text, high) <= max_tokens:
        raise AssertionError(
            f"Binary search error: high ({high}) tokens "
            f"({cached_encode_length(encoding_cache, encoding, text, high)}) "
            f"is not less than or equal to max_tokens"
        )
    if cached_encode_length(encoding_cache, encoding, text, low) > max_tokens:
        raise AssertionError(
            f"Binary search error: low ({low}) tokens "
            f"({cached_encode_length(encoding_cache, encoding, text, low)}) "
            f"is greater than max_tokens"
        )
    while low < high:
        mid = (low + high + 1) // 2
        mid_tokens = cached_encode_length(encoding_cache, encoding, text, mid)
        if mid_tokens <= max_tokens:
            if mid < low:
                raise AssertionError(
                    f"Binary search error: mid ({mid}) is not greater than low ({low}) - "
                    f"mid_tokens: {mid_tokens}, low: {low}, high: {high}"
                )
            elif mid == low:
                mid += 1
                if mid > high:
                    raise AssertionError(
                        f"Binary search error: mid ({mid}) is not greater than high ({high}) - "
                        f"mid_tokens: {mid_tokens}, low: {low}, high: {high}"
                    )
                low = mid
            else:
                if mid > high:
                    raise AssertionError(
                        f"Binary search error: mid ({mid}) is not less "
                        f"than high ({high}) - "
                        f"mid_tokens: {mid_tokens}, low: {low}, high: {high}"
                    )
                low = mid
        else:
            if mid > high:
                raise AssertionError(
                    f"Binary search error: mid ({mid}) is not less than "
                    f"high ({high}) - "
                    f"mid_tokens: {mid_tokens}, low: {low}, high: {high}"
                )
            elif mid == high:
                if mid_tokens == max_tokens:
                    return mid
                else:
                    if mid_tokens <= max_tokens:
                        raise AssertionError(
                            f"Binary search error: mid ({mid}) tokens "
                            f"({mid_tokens}) is not less than or equal to max_tokens"
                        )
                    high = mid - 1
            else:
                high = mid
    if low != high:
        raise AssertionError(f"Binary search error: low ({low}) is not equal to high ({high})")
    if cached_encode_length(encoding_cache, encoding, text, low) > max_tokens:
        raise AssertionError(
            f"Binary search error: low ({low}) tokens "
            f"({cached_encode_length(encoding_cache, encoding, text, low)}) "
            f"is not less than or equal to max_tokens {max_tokens}"
        )
    return low


@typechecked
def truncate_document_to_max_tokens(text: str, model: str) -> str:
    """Fast truncation of strings to the maximum token length, using tiktoken."""
    encoding_cache: Dict[int, int] = {}

    try:
        max_tokens = model_max_tokens[model]
        encoding = tiktoken.encoding_for_model(model)
        avg_tokens_per_char = get_avg_tokens_per_char(encoding)

        # Compute estimated_length based on average tokens per character
        estimated_length = int(max_tokens / avg_tokens_per_char)

        if estimated_length >= len(text):
            if cached_encode_length(encoding_cache, encoding, text, len(text)) <= max_tokens:
                return text  # Entire text is within limits, return as is
            else:
                high = len(text)
        else:
            high = estimated_length

        low = high

        high = expand_high(encoding_cache, encoding, text, high, max_tokens)
        if high == len(text):
            return text

        low = expand_low(encoding_cache, encoding, text, low, max_tokens)

        if cached_encode_length(encoding_cache, encoding, text, high) <= max_tokens:
            raise AssertionError(
                f"High bound error: unable to truncate text to {max_tokens} tokens, "
                f"current high: {high}, text length: {len(text)}"
            )

        max_length = binary_search_max_length(
            encoding_cache, encoding, text, low, high, max_tokens
        )

        return text[:max_length]

    finally:
        encoding_cache.clear()
