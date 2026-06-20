"""Tests for the self-learning DifficultyController."""

from __future__ import annotations

from umbra.difficulty import DifficultyController


def test_eases_when_below_target():
    c = DifficultyController(target_ssim=0.5, max_step=0.05)
    assert c.propose_step(0.2) < 0          # struggling -> ease off
    assert c.propose_step(0.2) >= -0.05     # capped


def test_pushes_when_above_target():
    c = DifficultyController(target_ssim=0.5, max_step=0.05)
    assert c.propose_step(0.8) > 0          # slack -> push harder
    assert c.propose_step(0.8) <= 0.05      # capped


def test_holds_at_target():
    c = DifficultyController(target_ssim=0.5)
    assert abs(c.propose_step(0.5)) < 1e-9


def test_learns_slope_from_observations():
    c = DifficultyController(slope_smoothing=0.0)  # adopt last measured slope
    c.observe(0.1, 0.8)
    c.observe(0.2, 0.6)                      # -0.2 SSIM over +0.1 difficulty -> slope -2
    assert c.learned_slope < -0.5
    assert abs(c.learned_slope - (-2.0)) < 0.5


def test_steps_capped_for_smoothness():
    c = DifficultyController(target_ssim=0.5, max_step=0.02)
    assert abs(c.propose_step(0.99)) <= 0.02
    assert abs(c.propose_step(0.0)) <= 0.02


def test_next_difficulty_stays_in_bounds():
    c = DifficultyController(max_step=0.5)
    assert 0.01 <= c.next_difficulty(0.02, 0.0) <= 1.0
    assert 0.01 <= c.next_difficulty(0.99, 0.9) <= 1.0


def test_state_roundtrip():
    c = DifficultyController(target_ssim=0.6, max_step=0.03)
    c.observe(0.1, 0.7)
    c.observe(0.2, 0.6)
    restored = DifficultyController.from_state(c.to_state())
    assert restored.target_ssim == 0.6
    assert abs(restored.learned_slope - c.learned_slope) < 1e-9
