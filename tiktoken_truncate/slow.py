"""Slow and safe implementation of truncate_document_to_max_tokens."""

import tiktoken
from typeguard import typechecked

from tiktoken_truncate.globals import model_max_tokens


@typechecked
def truncate_document_to_max_tokens(text: str, model: str) -> str:
    """Slow and safe implementation of truncate_document_to_max_tokens.

    Truncate the input text to a maximum number of tokens based on the specified model.

    Args:
        text (str): The input text to be truncated.
        model (str): The model to use for tokenization.

    Returns:
        str: The truncated text.

    Raises:
        AssertionError: If the text cannot be truncated to exactly the maximum number of tokens.
    """
    print(f"SLOW truncate_document_to_max_tokens: {len(text)} characters")

    max_tokens = model_max_tokens[model]
    encoding = tiktoken.encoding_for_model(model)

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        print(
            f"(slow) Text is already shorter than or equal to max_tokens "
            f"({len(tokens)} <= {max_tokens})"
        )
        return text

    for k in range(len(text), 1, -1):  # pragma: no cover
        text = text[:k]  # Remove the last character
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            break

    if len(tokens) > max_tokens:
        raise AssertionError("Cannot truncate text to exactly max_tokens")
    print(f"(slow) Truncated text to {len(tokens)} tokens using {k} characters")
    return text
