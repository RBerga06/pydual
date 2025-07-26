"""Package integrity test."""

from pathlib import Path


def test_integrity() -> None:
    """Package integrity test."""
    print("=> check[core]: import")
    import pydual.core

    print("=> check[core]: file py.typed")
    assert (Path(pydual.core.__file__).parent / "py.typed").exists()

    print("=> Everything looks good: enjoy!")
