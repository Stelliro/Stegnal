import zipfile

import numpy as np

from umbra.demo_packager import build_demo_executable


def test_build_demo_executable_creates_archive(tmp_path):
    image = np.zeros((16, 16, 3), dtype=np.float32)
    image[..., 0] = 1.0
    output = build_demo_executable(
        image,
        sample_rate=12_000,
        segments=2,
        marker_duration=0.02,
        label="Test candidate",
        metadata={"sound_score": 87.5},
        output_dir=tmp_path,
    )
    assert output.suffix == ".exe"
    assert output.exists()
    assert zipfile.is_zipfile(output)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "__main__.py" in names
        assert any(name.startswith("umbra/") for name in names)

