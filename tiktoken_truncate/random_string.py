"""Generate random strings for testing purposes."""

import random
import string

from typeguard import typechecked


@typechecked
def random_string(k: int, seed: int = 42) -> str:
    """Generate a random string of length n.

    Args:
        k (int): The length of the string to generate.
        seed (int, optional): The seed to use for the random number generator. Defaults to 42.

    Returns:
        str: The generated random string.
    """
    rng = random.Random()
    rng.seed(seed)
    return "".join(rng.choices(string.printable, k=k))
