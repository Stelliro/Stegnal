from umbra.testing import run_smoke_test


def test_smoke_test_runs_and_varies():
    metrics_a = run_smoke_test(seed=42, size=96)
    metrics_b = run_smoke_test(seed=123, size=128)

    assert 5.0 < metrics_a.psnr < 60.0
    assert 0.0 <= metrics_a.ssim <= 1.0
    assert metrics_b.psnr != metrics_a.psnr or metrics_b.ssim != metrics_a.ssim
