"""Tests for umbra.cli — argument parsing and command dispatch."""

from __future__ import annotations

import pytest

from umbra.cli import build_parser, main


def test_build_parser_returns_parser():
    parser = build_parser()
    assert parser is not None


def test_smoke_test_command_runs(capsys):
    """The smoke-test subcommand should run without errors."""
    main(["smoke-test", "--seed", "42", "--size", "32", "--sigma", "0.2", "--denoise-sigma", "0.5"])
    captured = capsys.readouterr()
    assert "Smoke test complete" in captured.out
    assert "PSNR" in captured.out


def test_encode_rejects_zero_sigma(tmp_path):
    img = tmp_path / "dummy.png"
    # create a tiny valid PNG
    from PIL import Image
    Image.fromarray((__import__("numpy").zeros((8, 8, 3), dtype="uint8"))).save(img)
    with pytest.raises(ValueError, match="positive"):
        main(["encode", "--image", str(img), "--output", str(tmp_path / "out.npz"), "--seed", "1", "--sigma", "0"])


def test_decode_rejects_negative_denoise_sigma(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        main([
            "decode", "--packet", str(tmp_path / "dummy.npz"),
            "--output", str(tmp_path / "out.png"),
            "--seed", "1", "--denoise-sigma", "-1",
        ])


def test_verbose_flag_accepted():
    """Ensure --verbose doesn't crash the parser."""
    parser = build_parser()
    args = parser.parse_args(["--verbose", "smoke-test"])
    assert args.verbose is True


def test_unknown_command_raises():
    with pytest.raises(SystemExit):
        main(["nonexistent-command"])
