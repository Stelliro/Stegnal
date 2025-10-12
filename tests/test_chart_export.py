from __future__ import annotations

import umbra.chart_export as chart_export


def test_export_chart_png_strips_params(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_png(spec: dict[str, object], **_: object) -> bytes:
        captured["spec"] = spec
        return b"png"

    monkeypatch.setattr(chart_export, "_VegaLite", None)
    monkeypatch.setattr(chart_export.vl_convert, "vegalite_to_png", fake_png)

    original = {"mark": "line", "params": [{"name": "history_view"}]}
    output = tmp_path / "chart.png"

    chart_export.export_chart_png(original, output)

    assert output.exists()
    assert original.get("params") is not None
    cleaned = captured["spec"]
    assert isinstance(cleaned, dict)
    assert "params" not in cleaned
