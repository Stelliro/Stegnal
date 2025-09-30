import numpy as np

from umbra.sound import generate_sound_art, guess_shapes


def test_generate_sound_art_deterministic_seed():
    color_a, gray_a, sound_a, shapes_a = generate_sound_art(seed=123, image_size=(96, 96))
    color_b, gray_b, sound_b, shapes_b = generate_sound_art(seed=123, image_size=(96, 96))

    assert np.allclose(color_a, color_b)
    assert np.allclose(gray_a, gray_b)
    assert sound_a.sample_rate == sound_b.sample_rate
    assert sound_a.band_volumes.keys() == sound_b.band_volumes.keys()


def test_guess_shapes_detects_channels_present():
    color, gray, sound, shapes = generate_sound_art(seed=7, image_size=(96, 96))
    guesses = guess_shapes(color)
    colors_present = {s.color for s in shapes}
    guess_colors = {g.color for g in guesses}
    assert colors_present.issubset(guess_colors)

