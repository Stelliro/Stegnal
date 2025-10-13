import importlib
import sys
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image

from umbra.evolution import EvolutionManager


def _install_ui_stubs(monkeypatch, state: dict[str, object]) -> None:
    class _StubFrame:
        def __init__(self, data: object) -> None:
            self._data = list(data) if data is not None else []
            self.empty = len(self._data) == 0

        def replace(self, *_args, **_kwargs):
            return self

        def dropna(self, *_args, **_kwargs):
            return self

        def to_csv(self, *_args, **_kwargs) -> str:
            return ""

    def _data_frame(data: object) -> _StubFrame:
        return _StubFrame(data)

    class _FakeConverter:
        def convert(self, _spec, **_kwargs):
            return b""

    class _FakeVegaLite:
        def __call__(self):
            return _FakeConverter()

    fake_pandas = SimpleNamespace(DataFrame=_data_frame)
    fake_vl_convert = SimpleNamespace(
        vl_convert=SimpleNamespace(vegalite_to_png=lambda _spec, **_opts: b""),
        VegaLite=_FakeVegaLite,
    )
    fake_streamlit = SimpleNamespace(
        session_state=state,
        experimental_rerun=lambda: None,
        rerun=lambda: None,
        sidebar=SimpleNamespace(info=lambda *_args, **_kwargs: None),
    )

    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)
    monkeypatch.setitem(sys.modules, "vl_convert", fake_vl_convert)


def test_ensure_manager_preserves_infinite_flag(monkeypatch) -> None:
    stub_state: dict[str, object] = {
        "run_infinite": False,
        "evolution_trees": {},
        "shared_seed": 123,
        "active_sound_seed": 99,
        "current_sound_sample_rate": 48_000,
        "current_sound_resolution": 128,
        "active_parent_seeds": [],
    }
    _install_ui_stubs(monkeypatch, stub_state)

    ui = importlib.import_module("umbra.ui")
    try:
        from umbra.decoding import NoiseStreamDecoder
        from umbra.encoding import NoiseStreamEncoder

        original = np.zeros((8, 8), dtype=np.float32)
        encoder = NoiseStreamEncoder(sigma=0.2)
        decoder = NoiseStreamDecoder(denoise_sigma=1.0)

        ui._ensure_manager(
            original,
            encoder,
            decoder,
            population_size=2,
            seed=123,
            autosave_interval=2,
        )

        assert stub_state["run_infinite"] is False

        stub_state["run_infinite"] = True
        stub_state["shared_seed"] = 456
        stub_state["active_sound_seed"] = 456

        ui._ensure_manager(
            original,
            encoder,
            decoder,
            population_size=2,
            seed=456,
            autosave_interval=2,
        )

        assert stub_state["run_infinite"] is True
    finally:
        sys.modules.pop("umbra.ui", None)


def test_auto_pause_does_not_cancel_infinite_mode(monkeypatch) -> None:
    stub_state: dict[str, object] = {
        "run_infinite": True,
        "pending_generations": 5,
        "evolution_mode": "Infinite",
    }
    _install_ui_stubs(monkeypatch, stub_state)

    ui = importlib.import_module("umbra.ui")
    try:
        message = ui._apply_auto_pause(
            stub_state,
            difficulty_progress=0.95,
            pause_threshold=0.9,
        )

        assert message == (
            "Difficulty spike detected – Keep Improving will continue running; "
            "refresh the scene manually if desired."
        )
        assert stub_state["run_infinite"] is True
        assert stub_state["pending_generations"] == 5
        assert stub_state["auto_pause_acknowledged"] is True
    finally:
        sys.modules.pop("umbra.ui", None)


def test_auto_pause_stops_finite_runs(monkeypatch) -> None:
    stub_state: dict[str, object] = {
        "run_infinite": False,
        "pending_generations": 3,
        "evolution_mode": "Finite",
    }
    _install_ui_stubs(monkeypatch, stub_state)

    ui = importlib.import_module("umbra.ui")
    try:
        message = ui._apply_auto_pause(
            stub_state,
            difficulty_progress=0.95,
            pause_threshold=0.9,
        )

        assert message == (
            "Difficulty spike reached – evolution paused so a new scene can be prepared."
        )
        assert stub_state["run_infinite"] is False
        assert stub_state["pending_generations"] == 0
        assert stub_state["auto_pause_acknowledged"] is True
    finally:
        sys.modules.pop("umbra.ui", None)


def test_auto_pause_resets_acknowledgement(monkeypatch) -> None:
    stub_state: dict[str, object] = {
        "run_infinite": False,
        "pending_generations": 0,
        "evolution_mode": "Finite",
        "auto_pause_acknowledged": True,
    }
    _install_ui_stubs(monkeypatch, stub_state)

    ui = importlib.import_module("umbra.ui")
    try:
        message = ui._apply_auto_pause(
            stub_state,
            difficulty_progress=0.5,
            pause_threshold=0.9,
        )

        assert message is None
        assert stub_state["auto_pause_acknowledged"] is False
    finally:
        sys.modules.pop("umbra.ui", None)


def test_normalize_pinterest_source(monkeypatch) -> None:
    _install_ui_stubs(monkeypatch, {})
    ui = importlib.import_module("umbra.ui")
    try:
        assert (
            ui._normalize_pinterest_source("umbra/research")
            == "https://www.pinterest.com/umbra/research.rss"
        )
        assert (
            ui._normalize_pinterest_source("https://example.com/feed.rss")
            == "https://example.com/feed.rss"
        )
        assert ui._normalize_pinterest_source("") in ui._PINTEREST_DEFAULT_FEEDS
    finally:
        sys.modules.pop("umbra.ui", None)


def test_parse_pinterest_feed_extracts_images(monkeypatch) -> None:
    _install_ui_stubs(monkeypatch, {})
    ui = importlib.import_module("umbra.ui")
    try:
        feed_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <rss version='2.0' xmlns:media='http://search.yahoo.com/mrss/'>
          <channel>
            <item>
              <title>Test Pin</title>
              <media:content url='https://i.pinimg.com/originals/example.jpg' />
              <description><![CDATA[<img src=\"https://i.pinimg.com/originals/example-desc.jpg\"/>]]></description>
            </item>
          </channel>
        </rss>
        """
        pairs = ui._parse_pinterest_feed(feed_xml)
        urls = {url for url, _ in pairs}
        assert "https://i.pinimg.com/originals/example.jpg" in urls
        assert "https://i.pinimg.com/originals/example-desc.jpg" in urls
        assert all(label for _, label in pairs)
    finally:
        sys.modules.pop("umbra.ui", None)


def test_parse_pinterest_feed_accepts_html(monkeypatch) -> None:
    _install_ui_stubs(monkeypatch, {})
    ui = importlib.import_module("umbra.ui")
    try:
        html_doc = r"""
        <html><head><title>Space Pins</title></head>
        <body>
          <script type="application/json">{"title":"Galaxy","images":{"orig":{"url":"https:\/\/i.pinimg.com\/originals\/galaxy.jpg"}}}</script>
          <img src="https://i.pinimg.com/736x/starfield.jpg" alt="Star Field" />
        </body>
        </html>
        """
        pairs = ui._parse_pinterest_feed(html_doc)
        urls = {url for url, _ in pairs}
        assert "https://i.pinimg.com/originals/galaxy.jpg" in urls
        assert "https://i.pinimg.com/736x/starfield.jpg" in urls
        assert any("Galaxy" in label for _, label in pairs)
        assert len(pairs) >= 2
    finally:
        sys.modules.pop("umbra.ui", None)


def test_fetch_random_pinterest_image_uses_downloader(monkeypatch) -> None:
    _install_ui_stubs(monkeypatch, {})
    ui = importlib.import_module("umbra.ui")
    try:
        with BytesIO() as buffer:
            Image.new("RGB", (2, 2), color=(200, 10, 10)).save(buffer, format="PNG")
            png_bytes = buffer.getvalue()

        feed_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <rss version='2.0' xmlns:media='http://search.yahoo.com/mrss/'>
          <channel>
            <item>
              <title>Sample Pin</title>
              <media:content url='https://i.pinimg.com/originals/sample.jpg' />
            </item>
          </channel>
        </rss>
        """

        requested: list[str] = []

        def fake_download(url: str, timeout: float) -> bytes:
            requested.append(url)
            if url.endswith(".rss"):
                return feed_xml.encode("utf-8")
            return png_bytes

        image, label = ui._fetch_random_pinterest_image(
            "umbra/research",
            timeout=0.1,
            download=fake_download,
        )

        assert image.shape == (2, 2, 3)
        assert label.endswith("(Pinterest)")
        assert any(url.endswith(".rss") for url in requested)
        assert any(url.endswith("sample.jpg") for url in requested)
    finally:
        sys.modules.pop("umbra.ui", None)


def test_auto_refresh_pinterest_reference_updates_state(monkeypatch) -> None:
    stub_state: dict[str, object] = {
        "quick_start_media_source": "pinterest",
        "quick_start_reference_source": "pinterest",
        "quick_start_pinterest_source": "space/inspiration",
    }

    _install_ui_stubs(monkeypatch, stub_state)
    ui = importlib.import_module("umbra.ui")

    try:
        def fake_fetch(source: str | None, *, timeout: float = 10.0, download=None):
            return np.ones((4, 4, 3), dtype=np.float32), "Galaxy (Pinterest)"

        monkeypatch.setattr(ui, "_fetch_random_pinterest_image", fake_fetch)

        refreshed = ui._auto_refresh_pinterest_reference(stub_state)
        assert refreshed is True
        assert np.array_equal(stub_state["quick_start_reference_image"], np.ones((4, 4, 3)))
        assert stub_state["quick_start_reference_label"] == "Galaxy (Pinterest)"
        assert stub_state["quick_start_reference_source"] == "pinterest"
        assert isinstance(stub_state.get("_last_pinterest_refresh"), float)
    finally:
        sys.modules.pop("umbra.ui", None)


def test_session_export_payload_contains_provenance(monkeypatch) -> None:
    from umbra.decoding import NoiseStreamDecoder
    from umbra.encoding import NoiseStreamEncoder
    from umbra.sound import generate_sound_art

    stub_state: dict[str, object] = {}
    _install_ui_stubs(monkeypatch, stub_state)

    from umbra.ui import _session_export_payload

    original = np.random.default_rng(0).random((16, 16), dtype=np.float32)
    encoder = NoiseStreamEncoder(sigma=0.2)
    decoder = NoiseStreamDecoder(denoise_sigma=0.9)
    manager = EvolutionManager(
        original=original,
        encoder=encoder,
        decoder=decoder,
        population_size=2,
        base_seed=2024,
        autosave_interval=2,
    )

    generation = manager.run_generation()
    best_candidate = generation.best_candidate

    state: dict[str, object] = {
        "hardware_backend": "CPU (NumPy)",
        "difficulty_progress": 0.2,
        "max_overlap_seen": float(best_candidate.overlap_score),
        "sound_reseed_count": 1,
        "difficulty_improvement": 0.1,
        "difficulty_volatility": 0.05,
        "performance_history": [],
        "latest_generation_difficulty": float(generation.difficulty_level),
        "latest_generation_difficulty_raw": float(generation.difficulty_raw),
    }

    _, _, sound_clip, _ = generate_sound_art(
        seed=manager.base_seed,
        image_size=(32, 32),
        sample_rate=44_100,
    )

    metrics = best_candidate.metrics
    payload = _session_export_payload(
        state=state,
        manager=manager,
        metrics=metrics,
        sound_metrics=metrics,
        sound_reference_metrics=metrics,
        ai_sound_alignment=metrics,
        ai_overlap_score=best_candidate.overlap_score,
        sound_overlap_score=best_candidate.overlap_score,
        sound_reference_overlap=best_candidate.overlap_score,
        sound_clip=sound_clip,
        base_encoder_sigma=0.2,
        base_decoder_sigma=1.0,
        encoder_sigma=0.2,
        denoise_sigma=0.9,
        current_sample_rate=sound_clip.sample_rate,
        current_resolution=32,
        sample_rate_range=(20_000, 48_000),
        resolution_range=(32, 64),
        seed=manager.base_seed,
        sound_seed=sound_clip.seed,
        target_dwell=5,
        best_candidate_summary={
            "generation": generation.index,
            "seed": best_candidate.seed,
            "psnr": best_candidate.metrics.psnr,
            "ssim": best_candidate.metrics.ssim,
            "overlap": best_candidate.overlap_score,
        },
    )

    assert "provenance" in payload
    assert "random_seeds" in payload["provenance"]
    metrics_block = payload["metrics"]
    assert "sound_vs_ai" in metrics_block
    assert "sound_vs_reference" in metrics_block
    assert "sound_vs_ai" in metrics_block["overlap"]
    assert metrics_block["global_pooled"]["desc"] == "pooled/global comparator"
    assert metrics_block["per_candidate_strict"]["desc"] == "gallery/best-candidate strict comparator"
    difficulty_block = payload["difficulty"]
    for key in ("raw", "normalized", "target"):
        assert key in difficulty_block
