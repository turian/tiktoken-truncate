"""Tests for `tiktoken_truncate` module."""

import random
from typing import Generator

import pytest
import tiktoken

import tiktoken_truncate
from tiktoken_truncate.globals import model_max_tokens
from tiktoken_truncate.random_string import random_string
from tiktoken_truncate.slow import (
    truncate_document_to_max_tokens as truncate_document_to_max_tokens_slow,
)
from tiktoken_truncate.tiktoken_truncate import (
    get_avg_tokens_per_char,
    truncate_document_to_max_tokens,
)

# Set up the test parameters
NTESTS = 30
rng = random.Random()
rng.seed(0)
models = list(model_max_tokens.keys())

# Generate test data
test_data = []
for _ in range(NTESTS):
    model = rng.choice(models)
    max_tokens = model_max_tokens[model]
    encoding = tiktoken.encoding_for_model(model)
    estimated_characters = max_tokens / get_avg_tokens_per_char(encoding=encoding)
    k = int(estimated_characters * rng.uniform(0.1, 2.0))
    text = random_string(k=k, seed=rng.randint(0, 2**32))
    test_data.append((model, text))


@pytest.mark.parametrize("model,text", test_data)
def test_slow_vs_fast(model: str, text: str) -> None:
    """Compare the slow and fast implementations of truncate_document_to_max_tokens."""
    text_slow = truncate_document_to_max_tokens_slow(text=text, model=model)
    text_fast = truncate_document_to_max_tokens(text=text, model=model)
    assert text_slow == text_fast


@pytest.fixture
def version() -> Generator[str, None, None]:
    """Sample pytest fixture."""
    yield tiktoken_truncate.__version__


def test_version(version: str) -> None:
    """Sample pytest test function with the pytest fixture as an argument."""
    assert version == "0.1.0"


if __name__ == "__main__":
    pytest.main()
