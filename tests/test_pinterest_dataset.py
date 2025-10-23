import numpy as np
from PIL import Image

from umbra.ui import PinterestDatasetEntry, PinterestDatasetManager


def _create_image(path, value: int) -> None:
    array = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def test_dataset_manager_cycles_and_archives(tmp_path) -> None:
    root = tmp_path / "datasets"
    manager = PinterestDatasetManager(
        root=root,
        feed_sources={},
        size_sequence=(64,),
        max_preview_pixels=128 * 128,
        pool_size=1,
        cycles_per_image=5,
        min_edge=1,
    )
    dataset_id = "manual"
    dataset_dir = root / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    image_path = dataset_dir / "image.png"
    _create_image(image_path, 32)
    entry = PinterestDatasetEntry(
        identifier="entry",
        url="https://example.com/image.png",
        label="Example",
        theme="unit-test",
        filename=image_path.name,
        size_bytes=image_path.stat().st_size,
        width=8,
        height=8,
    )
    manager._state = {
        "version": 1,
        "dataset_id": dataset_id,
        "entries": [entry.to_dict()],
        "rotation": {"queue": [entry.identifier], "index": 0},
        "used_urls": [],
    }
    manager._used_urls = set()
    manager._save_state()

    use_indices: list[int] = []
    for _ in range(5):
        acquisition = manager.acquire_image()
        use_indices.append(acquisition.use_index)
        assert acquisition.dataset_id == dataset_id
        assert acquisition.remaining_uses == max(0, 5 - acquisition.use_index)

    assert use_indices == [1, 2, 3, 4, 5]
    assert manager._state.get("dataset_id") is None
    assert manager._archive and manager._archive[-1]["dataset_id"] == dataset_id


def test_dataset_manager_respects_size_sequence(tmp_path) -> None:
    root = tmp_path / "datasets2"
    manager = PinterestDatasetManager(
        root=root,
        feed_sources={},
        size_sequence=(64, 128, 512),
        max_preview_pixels=512 * 512,
        pool_size=1,
        cycles_per_image=3,
        min_edge=1,
    )
    dataset_id = "sequence"
    dataset_dir = root / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    image_path = dataset_dir / "image.png"
    _create_image(image_path, 120)
    entry = PinterestDatasetEntry(
        identifier="entry-seq",
        url="https://example.com/image-seq.png",
        label="Sequence",
        theme="unit-test",
        filename=image_path.name,
        size_bytes=image_path.stat().st_size,
        width=8,
        height=8,
    )
    manager._state = {
        "version": 1,
        "dataset_id": dataset_id,
        "entries": [entry.to_dict()],
        "rotation": {"queue": [entry.identifier], "index": 0},
        "used_urls": [],
    }
    manager._used_urls = set()
    manager._save_state()

    declared_edges: list[int] = []
    actual_edges: list[int] = []
    for _ in range(3):
        acquisition = manager.acquire_image()
        declared_edges.append(acquisition.declared_edge)
        actual_edges.append(acquisition.actual_edge)

    assert declared_edges == [64, 128, 512]
    assert all(edge <= 512 for edge in actual_edges)
    assert manager._state.get("dataset_id") is None
