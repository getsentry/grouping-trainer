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
def test_strip_flags_and_their_values(argv: list[str], flags: tuple[str, ...], expected: list[str]) -> None:
    assert gt.launch._strip_flags_and_their_values(argv, flags) == expected


@pytest.mark.parametrize(
    "argv, flags, expected",
    [
        # bare flag is stripped, the following arg is NOT eaten (unlike _strip_flags_and_their_values)
        (["--sync_start", "--gpu", "h100"], ("--sync_start",), ["--gpu", "h100"]),
        # negation form is also stripped
        (["x", "--no_sync_start", "y"], ("--sync_start",), ["x", "y"]),
        # nothing to strip
        (["x", "y"], ("--sync_start",), ["x", "y"]),
        # not present
        (["--gpu", "h100"], ("--sync_start",), ["--gpu", "h100"]),
    ],
)
def test_strip_bool_flags(argv: list[str], flags: tuple[str, ...], expected: list[str]) -> None:
    assert gt.launch._strip_bool_flags(argv, flags) == expected


@pytest.mark.parametrize(
    "run_name, expected",
    [
        # canonical: single-token shortname
        ("2026-05-14-13-32-47-ddp", "ddp"),
        # shortname has internal hyphens — extra splits must stay in the shortname, not get lost
        ("2026-05-14-13-32-47-tiny-run", "tiny-run"),
        ("2026-05-14-13-32-47-multi-word-shortname", "multi-word-shortname"),
        # not in the expected timestamp format → return as-is
        ("not-a-run-name", "not-a-run-name"),
        ("", ""),
    ],
)
def test_shortname_from_run_name(run_name: str, expected: str) -> None:
    assert gt.launch.shortname_from_run_name(run_name) == expected


@pytest.mark.parametrize("shortname", ["ddp", "tiny-run", "multi-word-tag"])
def test_run_name_shortname_roundtrip(shortname: str) -> None:
    """`shortname_from_run_name` should recover the shortname from a freshly built run_name."""
    assert gt.launch.shortname_from_run_name(gt.launch.run_name_from_shortname(shortname)) == shortname
