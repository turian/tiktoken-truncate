"""Tests for `tiktoken_truncate` module."""

import random
from typing import Callable, Generator, List, Optional, Tuple

import pytest
import tiktoken

import tiktoken_truncate
from tiktoken_truncate.globals import model_max_tokens
from tiktoken_truncate.medium import (
    truncate_document_to_max_tokens as truncate_document_to_max_tokens_medium,
)
from tiktoken_truncate.random_string import random_string
from tiktoken_truncate.slow import (
    truncate_document_to_max_tokens as truncate_document_to_max_tokens_slow,
)
from tiktoken_truncate.tiktoken_truncate import get_avg_tokens_per_char
from tiktoken_truncate.tiktoken_truncate import (
    truncate_document_to_max_tokens as truncate_document_to_max_tokens_fast,
)

# Set up the test parameters
NTESTS_SLOW_VS_MEDIUM = 30
NTESTS_MEDIUM_VS_FAST = 100


def generate_test_data(
    ntests: int, seed: Optional[int], low_mult: float, high_mult: float
) -> List[Tuple[str, str]]:
    """Generate test data for given parameters.

    Args:
        ntests: Number of tests to generate.
        seed: Seed for random number generator (None for non-deterministic).
        low_mult: Lower bound for character estimation multiplier.
        high_mult: Upper bound for character estimation multiplier.

    Returns:
        A list of tuples containing model and text.
    """
    if seed:
        rng = random.Random(seed)
    else:
        # Non-deterministic version of tests
        rng = random.Random()
    models = list(model_max_tokens.keys())
    test_data = []
    for _ in range(ntests):
        model = rng.choice(models)
        max_tokens = model_max_tokens[model]
        encoding = tiktoken.encoding_for_model(model)
        estimated_characters = max_tokens / get_avg_tokens_per_char(encoding=encoding)
        k = int(estimated_characters * rng.uniform(low_mult, high_mult))
        text = random_string(k=k, seed=rng.randint(0, 2**32))
        test_data.append((model, text))
    return test_data


test_data_slow_vs_medium = generate_test_data(
    NTESTS_SLOW_VS_MEDIUM // 2, seed=0, low_mult=0.1, high_mult=2.0
)
test_data_slow_vs_medium = generate_test_data(
    NTESTS_SLOW_VS_MEDIUM // 2, seed=None, low_mult=0.1, high_mult=2.0
)
test_data_medium_vs_fast = generate_test_data(
    NTESTS_MEDIUM_VS_FAST // 2, seed=1, low_mult=0.01, high_mult=10.0
)
test_data_medium_vs_fast = generate_test_data(
    NTESTS_MEDIUM_VS_FAST // 2, seed=None, low_mult=0.01, high_mult=10.0
)


def run_comparison_test(
    impl1: Callable[[str, str], str],
    impl2: Callable[[str, str], str],
    model: str,
    text: str,
    label1: str,
    label2: str,
) -> None:
    """Compare the output of two implementations of truncate_document_to_max_tokens.

    Args:
        impl1: The first implementation to compare.
        impl2: The second implementation to compare.
        model: The model to use for tokenization.
        text: The text to be truncated.
        label1: Label for the first implementation.
        label2: Label for the second implementation.
    """
    text1 = impl1(text=text, model=model)
    text2 = impl2(text=text, model=model)
    assert text1 == text2, f"{label1} vs {label2} mismatch for model {model}"


@pytest.mark.parametrize("model,text", test_data_slow_vs_medium)
def test_slow_vs_medium(model: str, text: str) -> None:
    """Compare the slow and medium implementations of truncate_document_to_max_tokens."""
    run_comparison_test(
        truncate_document_to_max_tokens_slow,
        truncate_document_to_max_tokens_medium,
        model,
        text,
        "Slow",
        "Medium",
    )


@pytest.mark.parametrize("model,text", test_data_medium_vs_fast)
def test_medium_vs_fast(model: str, text: str) -> None:
    """Compare the medium and fast implementations of truncate_document_to_max_tokens."""
    run_comparison_test(
        truncate_document_to_max_tokens_medium,
        truncate_document_to_max_tokens_fast,
        model,
        text,
        "Medium",
        "Fast",
    )


@pytest.fixture
def version() -> Generator[str, None, None]:
    """Sample pytest fixture."""
    yield tiktoken_truncate.__version__


def test_version(version: str) -> None:
    """Sample pytest test function with the pytest fixture as an argument."""
    assert version == "0.1.0"


if __name__ == "__main__":
    pytest.main()
