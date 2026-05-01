import pytest

import grouping_trainer as gt


@pytest.mark.parametrize(
    "argv, flags, expected",
    [
        # space-separated form: --flag value
        (["--gpu", "l4", "--zone", "us-central1"], ("--gpu", "--zone"), []),
        # equals form: --flag=value
        (["--gpu=l4", "--zone=us-central1"], ("--gpu", "--zone"), []),
        # mixed: real args interleaved before, between, after
        (["x", "--gpu", "l4", "y"], ("--gpu",), ["x", "y"]),
        (["x", "--gpu=l4", "y"], ("--gpu",), ["x", "y"]),
        # only one of the two flags is present
        (["--keep", "v", "--gpu", "l4"], ("--gpu",), ["--keep", "v"]),
        # flag at the end with no following value — must not crash
        (["x", "--gpu"], ("--gpu",), ["x"]),
        # empty argv
        ([], ("--gpu",), []),
        # nothing to strip
        (["x", "y"], ("--gpu",), ["x", "y"]),
        # both forms in the same argv
        (["x", "--gpu", "l4", "--zone=us-east", "y"], ("--gpu", "--zone"), ["x", "y"]),
    ],
)
def test_strip_flags(argv: list[str], flags: tuple[str, ...], expected: list[str]) -> None:
    assert gt.launch._strip_flags(argv, flags) == expected
