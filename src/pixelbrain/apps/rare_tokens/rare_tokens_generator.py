from abc import ABC, abstractmethod
import os


class RareTokenGenerator(ABC):
    """Interface for a rare token generator."""

    def __iter__(self):
        """Return the iterator object."""
        return self

    @abstractmethod
    def __next__(self):
        """Return the next token in the order of rarity."""
        pass

class StableDiffution15RareTokenGenerator(RareTokenGenerator):
    """Generator that loads sd15_rare_tokens.txt and returns tokens in order."""

    def __init__(self):
        """Initialize the generator."""
        module_dir = os.path.dirname(os.path.abspath(__file__))
        tokens_file_path = os.path.join(module_dir, 'sd15_rare_tokens.txt')
        with open(tokens_file_path, 'r') as file:
            self.tokens = file.read().splitlines()
        self.index = 0

    def __next__(self) -> str:
        """Return the next token in the order of rarity."""
        if self.index >= len(self.tokens):
            raise StopIteration
        token = self.tokens[self.index]
        self.index += 1
        return token


def test_StableDiffution15RareTokenGenerator():
    generator = StableDiffution15RareTokenGenerator()
    tokens = []
    for _ in range(10):
        tokens.append(next(generator))
    expected_tokens = [
        'olis', 'lun', 'hta', 'dits', 'httr', 'rcn', 'sown', 'waj', 'shld', 'adl'
    ]
    assert expected_tokens == tokens