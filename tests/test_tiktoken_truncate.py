"""Tests for `tiktoken_truncate` module."""

import random
from typing import Generator, List, Optional, Protocol, Tuple

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
NTESTS_MEDIUM_VS_FAST = 1000


class TruncateFunction(Protocol):
    """Protocol for functions that truncate text to a maximum number of tokens.

    This protocol defines a callable that takes keyword arguments `text` and `model`,
    both of type `str`, and returns a `str`.
    """

    def __call__(self, *, text: str, model: str) -> str:
        """Dummy call."""
        ...


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
    if seed is not None:
        print(f"Using deterministic seed: {seed}")
        rng = random.Random(seed)
    else:
        # Non-deterministic version of tests
        seed = random.randint(0, 2**32 - 1)  # noqa: S311
        print(f"Using non-deterministic seed: {seed}")
        rng = random.Random(seed)
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
    impl1: TruncateFunction,
    impl2: TruncateFunction,
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
    assert text1 == text2, (
        f"{label1} vs {label2} mismatch for model {model}.\n"
        f"Original text: {text}\n"
        f"{label1} text: {text1}\n"
        f"{label2} text: {text2}\n"
        f"{label1} text length: {len(text1)}, "
        f"num tokens: {len(tiktoken.encoding_for_model(model).encode(text1))}\n"
        f"{label2} text length: {len(text2)}, "
        f"num tokens: {len(tiktoken.encoding_for_model(model).encode(text2))}\n"
    )


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


# Add explicit test cases with known inputs
explicit_test_cases = [
    ("text-embedding-3-large", "This is a simple test case for tiktoken truncate."),
    (
        "text-embedding-3-large",
        "Another example with a slightly longer text to test the truncation.",
    ),
    ("text-embedding-3-large", "A different model with a short text."),
    (
        "text-embedding-3-large",
        "Testing with a model and a much longer text to see how it handles.",
    ),
]


@pytest.mark.parametrize("model,text", explicit_test_cases)
def test_explicit_cases_slow_vs_medium(model: str, text: str) -> None:
    """Explicit test cases comparing the slow and medium implementations."""
    run_comparison_test(
        truncate_document_to_max_tokens_slow,
        truncate_document_to_max_tokens_medium,
        model,
        text,
        "Slow",
        "Medium",
    )


@pytest.mark.parametrize("model,text", explicit_test_cases)
def test_explicit_cases_medium_vs_fast(model: str, text: str) -> None:
    """Explicit test cases comparing the medium and fast implementations."""
    run_comparison_test(
        truncate_document_to_max_tokens_medium,
        truncate_document_to_max_tokens_fast,
        model,
        text,
        "Medium",
        "Fast",
    )


# Test function to run the comparison
def test_specific_input_medium_vs_fast() -> None:
    """Test specific input comparing the medium and fast implementations."""
    text = ".\x0bvO#-3U>{\\R;\t_K0uDL0T(U8*WT\\qO(#+r\\mD@D`{"

    run_comparison_test(
        truncate_document_to_max_tokens_medium,
        truncate_document_to_max_tokens_fast,
        "text-embedding-3-large",
        text,
        "Medium",
        "Fast",
    )


def test_version(version: str) -> None:
    """Sample pytest test function with the pytest fixture as an argument."""
    assert version == "0.1.0"


@pytest.fixture
def version() -> Generator[str, None, None]:
    """Sample pytest fixture."""
    yield tiktoken_truncate.__version__


if __name__ == "__main__":
    pytest.main()
